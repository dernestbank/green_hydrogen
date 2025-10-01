"""
User Feedback Collection and Analysis

Provides mechanisms to collect, analyze, and respond to user feedback
on system performance, usability, and requirements.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects user feedback on system performance and usage patterns.

    Features:
    - Performance feedback collection
    - Usability feedback tracking
    - System behavior monitoring
    - Anonymized usage analytics
    """

    def __init__(self, feedback_dir: str = "feedback"):
        """
        Initialize feedback collector.

        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)

        # Initialize feedback storage
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()

        # Load existing feedback if any
        self._load_feedback_data()

        logger.info(f"Feedback collector initialized with session {self.session_id}")

    def collect_performance_feedback(self, operation: str, metrics: Dict[str, Any],
                                    user_rating: Optional[int] = None,
                                    comments: Optional[str] = "") -> str:
        """
        Collect performance-related feedback.

        Args:
            operation: Name of operation that was performed
            metrics: Performance metrics (execution time, memory usage, etc.)
            user_rating: User satisfaction rating (1-5 scale)
            comments: User comments

        Returns:
            Feedback entry ID
        """
        feedback_id = str(uuid.uuid4())

        feedback_entry = {
            'id': feedback_id,
            'type': 'performance',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'metrics': metrics,
            'user_rating': user_rating,
            'comments': comments,
            'system_info': self._get_system_info()
        }

        # Store feedback
        self._store_feedback_entry(feedback_entry)

        logger.info(f"Performance feedback collected: {operation} (rating: {user_rating})")

        return feedback_id

    def collect_error_feedback(self, error_type: str, error_description: str,
                             context: Optional[Dict[str, Any]] = None,
                             recovery_successful: bool = False) -> str:
        """
        Collect error-related feedback.

        Args:
            error_type: Type of error encountered
            error_description: Description of the error
            context: Additional context information
            recovery_successful: Whether the system recovered successfully

        Returns:
            Feedback entry ID
        """
        feedback_id = str(uuid.uuid4())

        feedback_entry = {
            'id': feedback_id,
            'type': 'error',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_description': error_description,
            'context': context or {},
            'recovery_successful': recovery_successful,
            'system_info': self._get_system_info()
        }

        # Store feedback
        self._store_feedback_entry(feedback_entry)

        logger.error(f"Error feedback collected: {error_type}")

        return feedback_id

    def collect_usability_feedback(self, feature: str, ease_of_use: int,
                                 usefulness: int, suggestions: str = "") -> str:
        """
        Collect usability-related feedback.

        Args:
            feature: Feature being evaluated
            ease_of_use: Ease of use rating (1-5)
            usefulness: Usefulness rating (1-5)
            suggestions: User suggestions for improvement

        Returns:
            Feedback entry ID
        """
        feedback_id = str(uuid.uuid4())

        feedback_entry = {
            'id': feedback_id,
            'type': 'usability',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'feature': feature,
            'ease_of_use': ease_of_use,
            'usefulness': usefulness,
            'suggestions': suggestions,
            'system_info': self._get_system_info()
        }

        # Store feedback
        self._store_feedback_entry(feedback_entry)

        logger.info(f"Usability feedback collected: {feature} (ease: {ease_of_use}, useful: {usefulness})")

        return feedback_id

    def generate_feedback_summary(self) -> Dict[str, Any]:
        """
        Generate summary of collected feedback.

        Returns:
            Summary statistics and insights
        """
        summary = {
            'total_feedback_entries': len(self.feedback_data),
            'by_type': {},
            'avg_ratings': {},
            'error_rate': 0.0,
            'most_common_errors': [],
            'user_suggestions': [],
            'summary_timestamp': datetime.now().isoformat()
        }

        performance_ratings = []
        usability_ratings = []

        for entry in self.feedback_data.values():
            feedback_type = entry.get('type', 'unknown')

            # Count by type
            summary['by_type'][feedback_type] = summary['by_type'].get(feedback_type, 0) + 1

            # Collect ratings
            if feedback_type == 'performance' and entry.get('user_rating'):
                performance_ratings.append(entry['user_rating'])
            elif feedback_type == 'usability':
                usability_ratings.extend([entry.get('ease_of_use', 0), entry.get('usefulness', 0)])

            # Error tracking
            if feedback_type == 'error':
                summary['error_rate'] += 1

            # Collect suggestions
            if entry.get('suggestions'):
                summary['user_suggestions'].append(entry['suggestions'])

        # Calculate averages
        if performance_ratings:
            summary['avg_ratings']['performance'] = sum(performance_ratings) / len(performance_ratings)
        if usability_ratings:
            summary['avg_ratings']['usability'] = sum(usability_ratings) / len(usability_ratings)

        # Handle error rate
        total_operations = summary['by_type'].get('performance', 0)
        if total_operations > 0:
            summary['error_rate'] = summary['error_rate'] / total_operations

        logger.info(f"Feedback summary generated: {summary['total_feedback_entries']} entries")

        return summary

    def export_feedback_data(self, output_file: str = None) -> str:
        """
        Export feedback data to JSON file.

        Args:
            output_file: Optional output file path

        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"feedback_export_{timestamp}.json"

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_info': {
                'session_id': self.session_id,
                'session_start': self.session_start.isoformat(),
                'duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60
            },
            'feedback_data': self.feedback_data,
            'summary': self.generate_feedback_summary()
        }

        output_path = self.feedback_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Feedback data exported to {output_path}")
        return str(output_path)

    def _store_feedback_entry(self, entry: Dict[str, Any]):
        """Store feedback entry in memory."""
        entry_id = entry['id']
        self.feedback_data[entry_id] = entry

        # Also save to file for persistence
        feedback_file = self.feedback_dir / f"feedback_{entry_id[:8]}.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    def _load_feedback_data(self):
        """Load existing feedback data."""
        self.feedback_data = {}

        if self.feedback_dir.exists():
            for feedback_file in self.feedback_dir.glob("feedback_*.json"):
                try:
                    with open(feedback_file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                        self.feedback_data[entry['id']] = entry
                except Exception as e:
                    logger.warning(f"Failed to load feedback file {feedback_file}: {e}")

        logger.info(f"Loaded {len(self.feedback_data)} existing feedback entries")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            import psutil
            import platform

            return {
                'platform': platform.system(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': platform.python_version()
            }
        except ImportError:
            return {
                'platform': 'unknown',
                'error': 'psutil not available'
            }


class UsageAnalytics:
    """
    Track usage patterns and system behavior for continuous improvement.
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.operation_counts = {}
        self.error_counts = {}
        self.performance_metrics = []

    def track_operation(self, operation: str, duration: float):
        """
        Track operation usage and performance.

        Args:
            operation: Name of operation performed
            duration: Time taken in seconds
        """
        if operation not in self.operation_counts:
            self.operation_counts[operation] = 0
        self.operation_counts[operation] += 1

        self.performance_metrics.append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

    def track_error(self, error_type: str, error_message: str):
        """
        Track errors for analysis.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        logger.warning(f"Usage error tracked: {error_type} - {error_message}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of usage patterns.

        Returns:
            Usage summary statistics
        """
        total_operations = sum(self.operation_counts.values())

        return {
            'session_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'total_operations': total_operations,
            'operation_counts': self.operation_counts,
            'error_counts': self.error_counts,
            'average_operation_time': sum([m['duration'] for m in self.performance_metrics]) / len(self.performance_metrics) if self.performance_metrics else 0,
            'operations_per_minute': total_operations / max(1, (datetime.now() - self.start_time).total_seconds() / 60)
        }


# Global instances for convenience
feedback_collector = FeedbackCollector()
usage_analytics = UsageAnalytics()


def submit_performance_feedback(operation: str, execution_time: float,
                               memory_used: float, user_satisfaction: int = 3):
    """
    Convenience function to submit performance feedback.

    Args:
        operation: Operation that was performed
        execution_time: Time taken in seconds
        memory_used: Memory used in MB
        user_satisfaction: User satisfaction (1-5)
    """
    metrics = {
        'execution_time_seconds': execution_time,
        'memory_used_mb': memory_used
    }

    feedback_id = feedback_collector.collect_performance_feedback(
        operation=operation,
        metrics=metrics,
        user_rating=user_satisfaction
    )

    # Also track in usage analytics
    usage_analytics.track_operation(operation, execution_time)

    return feedback_id


def submit_error_feedback(error_type: str, error_description: str,
                         context: Dict[str, Any] = None):
    """
    Convenience function to submit error feedback.

    Args:
        error_type: Type of error
        error_description: Error description
        context: Additional context
    """
    feedback_id = feedback_collector.collect_error_feedback(
        error_type=error_type,
        error_description=error_description,
        context=context
    )

    # Also track in usage analytics
    usage_analytics.track_error(error_type, error_description)

    return feedback_id


def get_feedback_summary():
    """Get current feedback summary."""
    return feedback_collector.generate_feedback_summary()


def export_feedback_data(filename: str = None):
    """Export feedback data to file."""
    return feedback_collector.export_feedback_data(filename)