"""Circuit breaker pattern for external dependencies.

This module implements the circuit breaker pattern to prevent cascading failures
when external dependencies (like ML model loading) fail repeatedly.

Circuit states:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, reject requests immediately
- HALF_OPEN: Testing if service recovered, allow one request
"""

import logging
import time
import asyncio
from enum import Enum
from typing import Callable, Any, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    When failure threshold is reached, circuit opens and rejects
    requests immediately. After timeout, circuit enters half-open
    state to test if service recovered.

    Example:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        result = breaker.call(risky_function, arg1, arg2)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker entering HALF_OPEN state "
                    f"(failed {self.failure_count} times)"
                )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN (failed {self.failure_count} times). "
                    f"Will retry after {self.recovery_timeout}s timeout."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit breaker entering HALF_OPEN state "
                    f"(failed {self.failure_count} times)"
                )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN (failed {self.failure_count} times). "
                    f"Will retry after {self.recovery_timeout}s timeout."
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test recovery."""
        if self.last_failure_time is None:
            return True
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout

    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker test successful, closing circuit "
                f"(was open after {self.failure_count} failures)"
            )
            self.failure_count = 0
            self.success_count = 0

        self.state = CircuitState.CLOSED
        self.success_count += 1

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            old_state = self.state
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures "
                f"(was {old_state.value}). Will attempt reset after {self.recovery_timeout}s."
            )

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        old_state = self.state
        old_failures = self.failure_count
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info(
            f"Circuit breaker manually reset "
            f"(was {old_state.value} with {old_failures} failures)"
        )

    def get_state(self) -> dict:
        """
        Get current circuit breaker state.

        Returns:
            dict with state, failure_count, and time_since_last_failure
        """
        time_since_failure = None
        if self.last_failure_time:
            time_since_failure = time.time() - self.last_failure_time

        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_since_last_failure_seconds": time_since_failure,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejects requests."""
    pass


def circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """
    Decorator for circuit breaker pattern.

    Usage:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def risky_function():
            # ... code that might fail
            pass

        @circuit_breaker(failure_threshold=5, recovery_timeout=120)
        async def risky_async_function():
            # ... async code that might fail
            pass

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        expected_exception: Exception type to catch

    Returns:
        Decorated function with circuit breaker protection
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call_async(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            # Store breaker reference for testing/monitoring
            async_wrapper.circuit_breaker = breaker
            return async_wrapper
        else:
            # Store breaker reference for testing/monitoring
            sync_wrapper.circuit_breaker = breaker
            return sync_wrapper

    return decorator


# Module-level circuit breakers for ML model loading
# These persist across function calls to track failures
_whisper_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120,  # 2 minutes
    expected_exception=Exception
)

_pyannote_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120,  # 2 minutes
    expected_exception=Exception
)


def get_whisper_circuit_breaker() -> CircuitBreaker:
    """Get the shared Whisper model circuit breaker."""
    return _whisper_breaker


def get_pyannote_circuit_breaker() -> CircuitBreaker:
    """Get the shared Pyannote model circuit breaker."""
    return _pyannote_breaker
