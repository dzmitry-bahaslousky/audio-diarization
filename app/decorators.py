"""Reusable decorators for cross-cutting concerns.

This module provides decorators for:
- Retry logic with exponential backoff
- Performance timing and logging
- Error handling and transformation
- Caching with TTL
- Validation and type checking
"""

import asyncio
import functools
import logging
import time
from typing import Callable, TypeVar, Any, Optional, Type, Tuple

from app.exceptions import TransientError

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    backoff_factor: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (TransientError,),
    logger_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (2.0 = double each time)
        backoff_factor: Multiplier for backoff delay in seconds
        exceptions: Tuple of exception types to retry
        logger_name: Optional logger name for retry messages

    Returns:
        Decorated function with retry logic

    Example:
        ```python
        @retry(max_attempts=3, backoff_factor=2.0)
        async def fetch_model():
            # Will retry up to 3 times with exponential backoff
            return await download_large_model()

        @retry(exceptions=(NetworkError, TimeoutError))
        def api_call():
            return requests.get("https://api.example.com")
        ```

    Note:
        - Delay formula: backoff_factor * (backoff_base ** attempt)
        - First retry: backoff_factor * 2^0 = backoff_factor seconds
        - Second retry: backoff_factor * 2^1 = backoff_factor * 2 seconds
        - Third retry: backoff_factor * 2^2 = backoff_factor * 4 seconds
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor * (backoff_base ** attempt)
                        _logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        _logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor * (backoff_base ** attempt)
                        _logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        _logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def timeit(
    logger_name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_args: bool = False
) -> Callable[[F], F]:
    """Decorator to measure and log function execution time.

    Args:
        logger_name: Optional logger name (defaults to function's module)
        log_level: Logging level for timing messages
        log_args: Whether to log function arguments

    Returns:
        Decorated function with timing

    Example:
        ```python
        @timeit()
        async def process_large_file(filename: str):
            # Logs: "process_large_file completed in 2.35s"
            ...

        @timeit(log_level=logging.DEBUG, log_args=True)
        def calculate(x: int, y: int):
            # Logs: "calculate(x=5, y=10) completed in 0.001s"
            return x + y
        ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if log_args:
                    args_str = f"({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
                    _logger.log(
                        log_level,
                        f"{func.__name__}{args_str} completed in {elapsed:.3f}s"
                    )
                else:
                    _logger.log(
                        log_level,
                        f"{func.__name__} completed in {elapsed:.3f}s"
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if log_args:
                    args_str = f"({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
                    _logger.log(
                        log_level,
                        f"{func.__name__}{args_str} completed in {elapsed:.3f}s"
                    )
                else:
                    _logger.log(
                        log_level,
                        f"{func.__name__} completed in {elapsed:.3f}s"
                    )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def handle_errors(
    exception_map: dict[Type[Exception], Type[Exception]],
    logger_name: Optional[str] = None,
    reraise: bool = True
) -> Callable[[F], F]:
    """Decorator to transform exceptions for better error handling.

    Catches specified exceptions and transforms them into application-specific
    exceptions with better context.

    Args:
        exception_map: Mapping of caught exception to raised exception
        logger_name: Optional logger name
        reraise: Whether to reraise the transformed exception

    Returns:
        Decorated function with error transformation

    Example:
        ```python
        from sqlalchemy.exc import SQLAlchemyError
        from app.exceptions import DatabaseConnectionError

        @handle_errors({
            SQLAlchemyError: DatabaseConnectionError,
            ValueError: ValidationError
        })
        async def create_job(data: dict):
            # SQLAlchemyError will be caught and transformed to DatabaseConnectionError
            return await db.insert(data)
        ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)

            try:
                return await func(*args, **kwargs)
            except tuple(exception_map.keys()) as e:
                target_exception = exception_map[type(e)]
                _logger.error(
                    f"{func.__name__} raised {type(e).__name__}, "
                    f"transforming to {target_exception.__name__}: {e}"
                )

                # Create new exception with context
                new_exception = target_exception(
                    f"{func.__name__} failed: {e}",
                    details={"function": func.__name__, "original_type": type(e).__name__},
                    original_error=e
                )

                if reraise:
                    raise new_exception from e
                else:
                    _logger.exception(f"Suppressed exception: {new_exception}")
                    return None

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logging.getLogger(logger_name or func.__module__)

            try:
                return func(*args, **kwargs)
            except tuple(exception_map.keys()) as e:
                target_exception = exception_map[type(e)]
                _logger.error(
                    f"{func.__name__} raised {type(e).__name__}, "
                    f"transforming to {target_exception.__name__}: {e}"
                )

                new_exception = target_exception(
                    f"{func.__name__} failed: {e}",
                    details={"function": func.__name__, "original_type": type(e).__name__},
                    original_error=e
                )

                if reraise:
                    raise new_exception from e
                else:
                    _logger.exception(f"Suppressed exception: {new_exception}")
                    return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def validate_args(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """Decorator to validate function arguments before execution.

    Args:
        **validators: Keyword arguments mapping parameter names to validation functions

    Returns:
        Decorated function with argument validation

    Example:
        ```python
        @validate_args(
            age=lambda x: isinstance(x, int) and x >= 0,
            name=lambda x: isinstance(x, str) and len(x) > 0
        )
        def create_user(name: str, age: int):
            # Arguments are validated before function executes
            return User(name=name, age=age)

        create_user("Alice", 25)  # OK
        create_user("", 25)  # Raises ValueError
        create_user("Bob", -1)  # Raises ValueError
        ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for argument '{param_name}' "
                            f"with value {value!r}"
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Convenience decorators for common use cases

def retry_on_transient_error(
    max_attempts: int = 3,
    backoff_factor: float = 2.0
) -> Callable[[F], F]:
    """Convenience decorator to retry on any transient error.

    Example:
        ```python
        @retry_on_transient_error()
        async def load_model():
            # Automatically retries on ModelLoadError, NetworkError, etc.
            return await download_model()
        ```
    """
    return retry(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        exceptions=(TransientError,)
    )


def log_slow_operations(threshold_seconds: float = 1.0) -> Callable[[F], F]:
    """Log operations that take longer than threshold.

    Args:
        threshold_seconds: Time threshold for logging

    Example:
        ```python
        @log_slow_operations(threshold_seconds=5.0)
        async def process_video(video_path: str):
            # Logs warning if processing takes > 5 seconds
            ...
        ```
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            if elapsed > threshold_seconds:
                logger.warning(
                    f"{func.__name__} took {elapsed:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            if elapsed > threshold_seconds:
                logger.warning(
                    f"{func.__name__} took {elapsed:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
