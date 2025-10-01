"""
Comprehensive error handling and validation module.

Provides structured error handling, validation, and recovery mechanisms
for all components of the hydrogen production model.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import traceback
import functools
import time

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classifications."""
    VALIDATION = "validation"
    NETWORK = "network"
    FILE_SYSTEM = "filesystem"
    API = "api"
    CALCULATION = "calculation"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    TIMEOUT = "timeout"


@dataclass
class ErrorContext:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3
    context_data: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


class HydrogenModelError(Exception):
    """Base exception for hydrogen model errors."""

    def __init__(self, context: ErrorContext):
        self.context = context
        super().__init__(self.context.message)


class ValidationError(HydrogenModelError):
    """Validation-related errors."""
    pass


class NetworkError(HydrogenModelError):
    """Network and API-related errors."""
    pass


class CalculationError(HydrogenModelError):
    """Mathematical calculation errors."""
    pass


class ConfigurationError(HydrogenModelError):
    """Configuration-related errors."""
    pass


class ErrorHandler:
    """
    Centralized error handling and recovery manager.

    Provides comprehensive error handling, logging, and recovery strategies
    for all hydrogen model operations.
    """

    def __init__(self):
        self.error_log: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}

    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """
        Register a recovery strategy for a specific error type.

        Args:
            error_type: Type of error (e.g., 'validation', 'network')
            strategy: Recovery function to call
        """
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, context_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception that occurred
            context_data: Additional context information

        Returns:
            True if error was handled and recovered, False otherwise
        """
        # Determine error category and severity
        error_context = self._classify_error(error, context_data)

        # Log the error
        self._log_error(error_context)

        # Attempt recovery
        recovered = self._attempt_recovery(error_context)

        if not recovered:
            # Re-raise if recovery failed
            raise HydrogenModelError(error_context)

        return recovered

    def _classify_error(self, error: Exception, context_data: Optional[Dict[str, Any]]) -> ErrorContext:
        """Classify an error and create error context."""
        # Determine category
        if isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, TimeoutError, OSError)):
            category = ErrorCategory.NETWORK
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            category = ErrorCategory.FILE_SYSTEM
        elif isinstance(error, KeyError):
            category = ErrorCategory.CONFIGURATION
        elif isinstance(error, (ZeroDivisionError, OverflowError, ArithmeticError)):
            category = ErrorCategory.CALCULATION
        elif isinstance(error, RecursionError):
            category = ErrorCategory.MEMORY
        else:
            category = ErrorCategory.CALCULATION

        # Determine severity
        severity = self._determine_severity(category, error)

        # Get traceback
        tb = traceback.format_exc()

        return ErrorContext(
            category=category,
            severity=severity,
            message=str(error),
            details=tb,
            recoverable=self._is_recoverable(category),
            context_data=context_data,
            traceback=tb
        )

    def _determine_severity(self, category: ErrorCategory, error: Exception) -> ErrorSeverity:
        """Determine error severity based on category and error type."""
        if category in [ErrorCategory.CALCULATION, ErrorCategory.NETWORK]:
            if "critical" in str(error).lower():
                return ErrorSeverity.CRITICAL
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.VALIDATION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.FILE_SYSTEM:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.LOW

    def _is_recoverable(self, category: ErrorCategory) -> bool:
        """Determine if an error category is recoverable."""
        return category in [
            ErrorCategory.NETWORK,
            ErrorCategory.VALIDATION,
            # FILE_SYSTEM and CONFIGURATION errors can sometimes be recovered
        ]

    def _log_error(self, context: ErrorContext):
        """Log error with appropriate level based on severity."""
        self.error_log.append(context)

        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error ({context.category.value}): {context.message}")
        elif context.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]:
            logger.error(f"Error ({context.category.value}): {context.message}")
        else:
            logger.warning(f"Warning ({context.category.value}): {context.message}")

        if context.details:
            logger.debug(f"Error details: {context.details}")

    def _attempt_recovery(self, context: ErrorContext) -> bool:
        """Attempt to recover from an error."""
        if not context.recoverable:
            return False

        recovery_key = f"{context.category.value}_{context.severity.value}"
        strategy = self.recovery_strategies.get(recovery_key)

        if strategy and context.retry_count < context.max_retries:
            try:
                logger.info(f"Attempting recovery for {recovery_key} (attempt {context.retry_count + 1})")
                result = strategy(context)
                if result:
                    logger.info(f"Recovery successful for {recovery_key}")
                    return True
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {recovery_key}: {recovery_error}")
                context.retry_count += 1
                # Recursively attempt recovery (but limit recursion)
                if context.retry_count < context.max_retries:
                    return self._attempt_recovery(context)

        return False

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all logged errors."""
        summary = {
            'total_errors': len(self.error_log),
            'by_category': {},
            'by_severity': {},
            'critical_errors': [],
            'unrecovered_errors': []
        }

        for error in self.error_log:
            cat, sev = error.category.value, error.severity.value

            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
            summary['by_severity'][sev] = summary['by_severity'].get(sev, 0) + 1

            if error.severity == ErrorSeverity.CRITICAL:
                summary['critical_errors'].append(str(error))

            if not error.recoverable:
                summary['unrecovered_errors'].append(str(error))

        return summary


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(category: ErrorCategory = ErrorCategory.CALCULATION,
                        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                        max_retries: int = 3):
    """
    Decorator for comprehensive error handling.

    Args:
        category: Error category
        severity: Error severity
        max_retries: Maximum retry attempts
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HydrogenModelError:
                # Re-raise hydrogen model errors
                raise
            except Exception as error:
                logger.error(f"Unhandled error in {func.__name__}: {error}")

                # Create error context
                context = ErrorContext(
                    category=category,
                    severity=severity,
                    message=str(error),
                    details=traceback.format_exc(),
                    max_retries=max_retries,
                    recoverable=severity != ErrorSeverity.CRITICAL
                )

                # Handle the error
                if not error_handler.handle_error(error, {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }):
                    raise HydrogenModelError(context)

        return wrapper
    return decorator


@contextmanager
def error_context(description: str,
                 category: ErrorCategory = ErrorCategory.CALCULATION):
    """
    Context manager for error handling in complex operations.

    Args:
        description: Description of the operation
        category: Error category
    """
    try:
        logger.debug(f"Entering error context: {description}")
        yield
    except Exception as error:
        logger.error(f"Error in context '{description}': {error}")
        error_handler.handle_error(error, {'context': description, 'category': category.value})
        raise
    finally:
        logger.debug(f"Exiting error context: {description}")


def validate_numeric_input(value: Any, min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          value_name: str = "value") -> float:
    """
    Validate numeric input with range checking.

    Args:
        value: Input value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        value_name: Name of the value for error messages

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                message=f"{value_name} must be numeric, got {type(value).__name__}",
                context_data={'value': value, 'value_name': value_name}
            )
        )

    if min_val is not None and numeric_value < min_val:
        raise ValidationError(
            ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                message=f"{value_name} must be >= {min_val}, got {numeric_value}",
                context_data={'value': numeric_value, 'min_val': min_val, 'value_name': value_name}
            )
        )

    if max_val is not None and numeric_value > max_val:
        raise ValidationError(
            ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                message=f"{value_name} must be <= {max_val}, got {numeric_value}",
                context_data={'value': numeric_value, 'max_val': max_val, 'value_name': value_name}
            )
        )

    return numeric_value


def validate_configuration(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary against schema.

    Args:
        config: Configuration dictionary
        schema: Schema dictionary

    Returns:
        True if valid, False otherwise

    Raises:
        ConfigurationError: If validation fails
    """
    # Simple schema validation - can be extended for complex schemes
    errors = []

    for key, spec in schema.items():
        if key not in config and spec.get('required', False):
            errors.append(f"Required configuration key missing: {key}")
        elif key in config:
            value = config[key]
            expected_type = spec.get('type')

            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Configuration key '{key}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}")

            # Range checking if specified
            if 'min' in spec and value < spec['min']:
                errors.append(f"Configuration key '{key}' must be >= {spec['min']}, got {value}")

            if 'max' in spec and value > spec['max']:
                errors.append(f"Configuration key '{key}' must be <= {spec['max']}, got {value}")

    if errors:
        raise ConfigurationError(
            ErrorContext(
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                message=f"Configuration validation failed: {'; '.join(errors)}",
                context_data={'validation_errors': errors, 'config': config}
            )
        )

    return True


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0,
                      max_delay: float = 30.0, backoff_factor: float = 2.0):
    """
    Retry decorator with exponential backoff for network operations.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Exponential backoff factor
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        break

                    delay = min(delay * backoff_factor, max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, "
                                 f"retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# Register common recovery strategies
def network_recovery(context: ErrorContext) -> bool:
    """Recovery strategy for network errors."""
    # Simple delay and retry
    time.sleep(1.0)
    return True

error_handler.register_recovery_strategy("network_medium", network_recovery)