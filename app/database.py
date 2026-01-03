"""Database connection and session management with connection pooling and health checks.

This module provides:
- SQLAlchemy engine configuration with optimal pool settings
- Session lifecycle management via context managers
- Repository pattern support with automatic session handling
- Database health checking and initialization
"""

from typing import Generator, TypeVar, Type, Optional
from sqlalchemy import create_engine, text, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool
from contextlib import contextmanager
import logging
import time

from app.config import settings
from app.models.database import Base

logger = logging.getLogger(__name__)

# Type variable for repository classes
T = TypeVar('T')


# Event listeners for connection pool monitoring
@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log database connections for monitoring."""
    logger.debug("Database connection established")


@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection pool checkouts."""
    logger.debug("Connection checked out from pool")


@event.listens_for(Engine, "before_cursor_execute", named=True)
def receive_before_cursor_execute(**kw):
    """Store query execution start time for performance monitoring."""
    kw['connection_proxy']._query_start_time = time.perf_counter()


@event.listens_for(Engine, "after_cursor_execute", named=True)
def receive_after_cursor_execute(**kw):
    """Log slow queries for performance analysis."""
    connection_proxy = kw['connection_proxy']
    if hasattr(connection_proxy, '_query_start_time'):
        elapsed = time.perf_counter() - connection_proxy._query_start_time
        if elapsed > 1.0:  # Log queries taking more than 1 second
            logger.warning(
                f"Slow query detected: {elapsed:.2f}s - {kw['statement']}"
            )


# Create engine with optimized settings
engine = create_engine(
    settings.database_url,
    pool_size=5,  # Core pool size
    max_overflow=10,  # Additional connections under load
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=30,  # Max wait time for connection from pool
    echo=settings.log_level == "DEBUG",
    echo_pool=settings.log_level == "DEBUG",
    future=True  # Use SQLAlchemy 2.0 style
)

# Session factory with optimal settings
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=True  # Refresh objects after commit
)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes with automatic session management.

    Provides a database session that is automatically committed on success
    and rolled back on error. The session is always closed properly.

    Usage:
        ```python
        from fastapi import Depends

        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session here
            job = db.query(TranscriptionJob).first()
        ```

    Yields:
        SQLAlchemy session

    Note:
        FastAPI's dependency injection ensures this generator is properly
        cleaned up even if the endpoint raises an exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions in non-FastAPI code.

    Provides a session with automatic cleanup. Use this in Celery tasks,
    CLI scripts, or any code outside of FastAPI request handlers.

    Usage:
        ```python
        with get_db_context() as db:
            job = db.query(TranscriptionJob).first()
            # Session automatically closed on context exit
        ```

    Yields:
        SQLAlchemy session

    Example:
        ```python
        # In a Celery task
        def process_job(job_id: str):
            with get_db_context() as db:
                job = db.query(TranscriptionJob).filter_by(id=job_id).first()
                if job:
                    job.status = "processing"
                    db.commit()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_repository(
    repository_class: Type[T]
) -> Generator[T, None, None]:
    """Context manager for repositories with automatic session management.

    This creates a repository instance with a database session and ensures
    the session is properly closed after use. Repositories encapsulate all
    database operations for a specific entity type.

    Args:
        repository_class: Repository class to instantiate (must accept Session in __init__)

    Usage:
        ```python
        from app.repositories.transcription_repository import TranscriptionRepository

        with get_repository(TranscriptionRepository) as repo:
            job = repo.create_job(
                filename="audio.mp3",
                upload_path="/path/to/file",
                whisper_model="medium"
            )
        ```

    Yields:
        Repository instance with active database session

    Note:
        The repository should NOT commit transactions itself - use the
        repository's transaction() context manager for explicit control.
    """
    db = SessionLocal()
    try:
        yield repository_class(db)
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables using SQLAlchemy metadata.

    Creates all tables defined in models if they don't exist.
    Safe to call multiple times - won't recreate existing tables.

    Note:
        This is a simple initialization suitable for development.
        Production deployments should use proper migrations (Alembic).

    Raises:
        Exception: If table creation fails
    """
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}", exc_info=True)
        raise


def check_db_connection(timeout: float = 5.0) -> bool:
    """Check if database is accessible with timeout.

    Attempts a simple query to verify database connectivity.
    Useful for health checks and startup validation.

    Args:
        timeout: Maximum time to wait for connection in seconds

    Returns:
        True if database is accessible, False otherwise

    Example:
        ```python
        if not check_db_connection():
            logger.error("Database unavailable, exiting")
            sys.exit(1)
        ```
    """
    try:
        start_time = time.perf_counter()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        elapsed = time.perf_counter() - start_time
        logger.debug(f"Database connection check passed ({elapsed:.3f}s)")
        return True

    except Exception as e:
        logger.error(f"Database connection failed: {e}", exc_info=True)
        return False


def get_db_stats() -> dict:
    """Get database connection pool statistics.

    Returns detailed metrics about the connection pool state,
    useful for monitoring and debugging connection issues.

    Returns:
        Dictionary with pool statistics

    Example:
        ```python
        stats = get_db_stats()
        logger.info(f"Pool size: {stats['pool_size']}, "
                   f"Checked out: {stats['checked_out']}")
        ```
    """
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "checked_in": pool.checkedin(),
        "total_connections": pool.size() + pool.overflow(),
    }


def close_db() -> None:
    """Close all database connections and dispose of the engine.

    Call this during application shutdown to cleanly release
    all database resources.

    Example:
        ```python
        # In FastAPI lifespan shutdown
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            close_db()
        ```
    """
    logger.info("Closing database connections...")
    engine.dispose()
    logger.info("Database connections closed")
