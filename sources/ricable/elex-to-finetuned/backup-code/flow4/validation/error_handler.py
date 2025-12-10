"""
Comprehensive error handling and recovery system for Flow4.

This module provides:
- Error detection and classification
- Automatic recovery procedures
- Error reporting and logging
- System health monitoring
"""

import os
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILE_SYSTEM = "file_system"
    CONVERSION = "conversion"
    CHUNKING = "chunking"
    DATASET_GENERATION = "dataset_generation"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    MEMORY = "memory"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: str
    category: ErrorCategory
    severity: ErrorSeverity
    description: str
    context: Dict[str, Any]
    traceback_info: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """Definition of a recovery action."""
    name: str
    description: str
    action_func: Callable
    applicable_categories: List[ErrorCategory]
    max_attempts: int = 3
    timeout: int = 60


class ErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self, workspace_dir: str = "pipeline_output"):
        """Initialize error handler.
        
        Args:
            workspace_dir: Main workspace directory
        """
        self.workspace_dir = Path(workspace_dir)
        self.error_log = []
        self.recovery_actions = self._initialize_recovery_actions()
        self.health_status = {}
        
        # Create error reporting directory
        self.error_dir = self.workspace_dir / "error_reports"
        self.error_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_recovery_actions(self) -> List[RecoveryAction]:
        """Initialize available recovery actions."""
        return [
            RecoveryAction(
                name="cleanup_temp_files",
                description="Clean up temporary files and directories",
                action_func=self._cleanup_temp_files,
                applicable_categories=[ErrorCategory.FILE_SYSTEM, ErrorCategory.MEMORY]
            ),
            RecoveryAction(
                name="recreate_output_structure",
                description="Recreate missing output directory structure",
                action_func=self._recreate_output_structure,
                applicable_categories=[ErrorCategory.FILE_SYSTEM, ErrorCategory.CONFIGURATION]
            ),
            RecoveryAction(
                name="check_dependencies",
                description="Verify and attempt to fix dependency issues",
                action_func=self._check_dependencies,
                applicable_categories=[ErrorCategory.DEPENDENCY]
            ),
            RecoveryAction(
                name="reset_corrupted_files",
                description="Remove corrupted files to allow regeneration",
                action_func=self._reset_corrupted_files,
                applicable_categories=[ErrorCategory.CONVERSION, ErrorCategory.CHUNKING, ErrorCategory.DATASET_GENERATION]
            ),
            RecoveryAction(
                name="free_memory",
                description="Attempt to free system memory",
                action_func=self._free_memory,
                applicable_categories=[ErrorCategory.MEMORY]
            ),
            RecoveryAction(
                name="restart_services",
                description="Restart relevant services or components",
                action_func=self._restart_services,
                applicable_categories=[ErrorCategory.MODEL_TRAINING, ErrorCategory.NETWORK]
            )
        ]

    def handle_error(self, 
                    exception: Exception, 
                    context: Dict[str, Any],
                    attempt_recovery: bool = True) -> ErrorRecord:
        """Handle an error occurrence.
        
        Args:
            exception: The exception that occurred
            context: Context information about where the error occurred
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Error record with recovery information
        """
        # Generate unique error ID
        error_id = f"err_{int(time.time() * 1000)}"
        
        # Classify the error
        category = self._classify_error(exception, context)
        severity = self._assess_severity(exception, context, category)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            category=category,
            severity=severity,
            description=str(exception),
            context=context,
            traceback_info=traceback.format_exc()
        )
        
        # Log the error
        logger.error(f"Error {error_id}: {exception}")
        logger.debug(f"Error context: {context}")
        
        # Attempt recovery if requested and appropriate
        if attempt_recovery and severity != ErrorSeverity.CRITICAL:
            self._attempt_recovery(error_record)
        
        # Save error record
        self.error_log.append(error_record)
        self._save_error_record(error_record)
        
        # Update health status
        self._update_health_status(error_record)
        
        return error_record

    def _classify_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__
        
        # File system errors
        if isinstance(exception, (FileNotFoundError, PermissionError, OSError)):
            return ErrorCategory.FILE_SYSTEM
        
        # Memory errors
        if isinstance(exception, MemoryError) or 'memory' in error_message:
            return ErrorCategory.MEMORY
        
        # Network errors
        if 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        
        # Dependency errors
        if isinstance(exception, ImportError) or 'module' in error_message:
            return ErrorCategory.DEPENDENCY
        
        # Context-based classification
        phase = context.get('phase', '').lower()
        if 'conversion' in phase:
            return ErrorCategory.CONVERSION
        elif 'chunking' in phase:
            return ErrorCategory.CHUNKING
        elif 'dataset' in phase or 'augmentoolkit' in phase:
            return ErrorCategory.DATASET_GENERATION
        elif 'training' in phase or 'finetune' in phase:
            return ErrorCategory.MODEL_TRAINING
        elif 'validation' in phase:
            return ErrorCategory.VALIDATION
        elif 'config' in phase:
            return ErrorCategory.CONFIGURATION
        
        # Component-based classification
        component = context.get('component', '').lower()
        if 'docling' in component or 'converter' in component:
            return ErrorCategory.CONVERSION
        elif 'chunker' in component:
            return ErrorCategory.CHUNKING
        elif 'mlx' in component or 'model' in component:
            return ErrorCategory.MODEL_TRAINING
        
        return ErrorCategory.UNKNOWN

    def _assess_severity(self, exception: Exception, context: Dict[str, Any], category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity."""
        # Critical errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        if isinstance(exception, MemoryError):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.DEPENDENCY, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.HIGH
        
        if 'critical' in str(exception).lower():
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.CONVERSION, ErrorCategory.MODEL_TRAINING]:
            return ErrorSeverity.MEDIUM
        
        if isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW

    def _attempt_recovery(self, error_record: ErrorRecord):
        """Attempt to recover from the error."""
        error_record.recovery_attempted = True
        
        # Find applicable recovery actions
        applicable_actions = [
            action for action in self.recovery_actions
            if error_record.category in action.applicable_categories
        ]
        
        if not applicable_actions:
            logger.info(f"No recovery actions available for {error_record.category}")
            return
        
        # Sort by priority (based on success rate, custom logic, etc.)
        applicable_actions.sort(key=lambda x: x.name)
        
        for action in applicable_actions:
            logger.info(f"Attempting recovery action: {action.name}")
            
            try:
                success = action.action_func(error_record)
                
                if success:
                    error_record.recovery_successful = True
                    error_record.recovery_actions.append(f"✓ {action.name}")
                    logger.info(f"Recovery action '{action.name}' succeeded")
                    break
                else:
                    error_record.recovery_actions.append(f"✗ {action.name}")
                    logger.warning(f"Recovery action '{action.name}' failed")
            
            except Exception as e:
                error_record.recovery_actions.append(f"✗ {action.name} (error: {str(e)})")
                logger.error(f"Recovery action '{action.name}' raised exception: {e}")

    def _cleanup_temp_files(self, error_record: ErrorRecord) -> bool:
        """Clean up temporary files and directories."""
        try:
            # Clean up system temp directory
            temp_dir = Path(tempfile.gettempdir())
            flow4_temp_files = list(temp_dir.glob("flow4_*"))
            
            for temp_file in flow4_temp_files:
                try:
                    if temp_file.is_dir():
                        shutil.rmtree(temp_file)
                    else:
                        temp_file.unlink()
                except Exception:
                    pass
            
            # Clean up workspace temp files
            workspace_temp = self.workspace_dir / "temp"
            if workspace_temp.exists():
                shutil.rmtree(workspace_temp)
            
            logger.info(f"Cleaned up {len(flow4_temp_files)} temporary files")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clean up temp files: {e}")
            return False

    def _recreate_output_structure(self, error_record: ErrorRecord) -> bool:
        """Recreate missing output directory structure."""
        try:
            # Standard Flow4 directory structure
            directories = [
                "extracted",
                "markdown",
                "combined",
                "chunks",
                "rag_datasets",
                "augmentoolkit_datasets",
                "validation_reports"
            ]
            
            for dir_name in directories:
                dir_path = self.workspace_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Recreated output directory structure")
            return True
        
        except Exception as e:
            logger.error(f"Failed to recreate directory structure: {e}")
            return False

    def _check_dependencies(self, error_record: ErrorRecord) -> bool:
        """Check and attempt to fix dependency issues."""
        try:
            # Check key dependencies
            dependencies = {
                'docling': 'Document conversion functionality',
                'mlx': 'MLX fine-tuning functionality',
                'mlx_lm': 'MLX language model functionality'
            }
            
            missing_deps = []
            
            for dep_name, description in dependencies.items():
                try:
                    __import__(dep_name)
                except ImportError:
                    missing_deps.append((dep_name, description))
            
            if missing_deps:
                logger.warning(f"Missing dependencies: {[dep[0] for dep in missing_deps]}")
                # Could potentially suggest installation commands
                return False
            
            logger.info("All key dependencies are available")
            return True
        
        except Exception as e:
            logger.error(f"Failed to check dependencies: {e}")
            return False

    def _reset_corrupted_files(self, error_record: ErrorRecord) -> bool:
        """Remove corrupted files to allow regeneration."""
        try:
            context = error_record.context
            
            # Check for file paths in context
            file_paths = []
            for key, value in context.items():
                if isinstance(value, (str, Path)) and 'path' in key.lower():
                    file_paths.append(Path(value))
            
            removed_count = 0
            for file_path in file_paths:
                if file_path.exists():
                    try:
                        if file_path.is_dir():
                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()
                        removed_count += 1
                        logger.info(f"Removed potentially corrupted file: {file_path}")
                    except Exception:
                        pass
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} potentially corrupted files")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to reset corrupted files: {e}")
            return False

    def _free_memory(self, error_record: ErrorRecord) -> bool:
        """Attempt to free system memory."""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear any large objects if possible
            # This is a placeholder for more sophisticated memory management
            
            logger.info(f"Garbage collection freed {collected} objects")
            return True
        
        except Exception as e:
            logger.error(f"Failed to free memory: {e}")
            return False

    def _restart_services(self, error_record: ErrorRecord) -> bool:
        """Restart relevant services or components."""
        try:
            # This is a placeholder for service restart logic
            # In a real implementation, this might restart specific components
            
            logger.info("Service restart placeholder - no actual restart performed")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restart services: {e}")
            return False

    def _save_error_record(self, error_record: ErrorRecord):
        """Save error record to file."""
        try:
            error_file = self.error_dir / f"error_{error_record.error_id}.json"
            
            # Convert to serializable format
            record_data = {
                'error_id': error_record.error_id,
                'timestamp': error_record.timestamp,
                'category': error_record.category.value,
                'severity': error_record.severity.value,
                'description': error_record.description,
                'context': error_record.context,
                'traceback_info': error_record.traceback_info,
                'recovery_attempted': error_record.recovery_attempted,
                'recovery_successful': error_record.recovery_successful,
                'recovery_actions': error_record.recovery_actions
            }
            
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(record_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Failed to save error record: {e}")

    def _update_health_status(self, error_record: ErrorRecord):
        """Update system health status based on error."""
        category = error_record.category.value
        
        if category not in self.health_status:
            self.health_status[category] = {
                'error_count': 0,
                'last_error': None,
                'recovery_success_rate': 0.0
            }
        
        # Update error count
        self.health_status[category]['error_count'] += 1
        self.health_status[category]['last_error'] = error_record.timestamp
        
        # Update recovery success rate
        category_errors = [e for e in self.error_log if e.category == error_record.category]
        recovery_attempts = [e for e in category_errors if e.recovery_attempted]
        
        if recovery_attempts:
            successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
            self.health_status[category]['recovery_success_rate'] = len(successful_recoveries) / len(recovery_attempts)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and system health."""
        if not self.error_log:
            return {
                'total_errors': 0,
                'health_status': 'good',
                'message': 'No errors recorded'
            }
        
        # Calculate summary statistics
        total_errors = len(self.error_log)
        recent_errors = [e for e in self.error_log if 
                        (datetime.now() - datetime.fromisoformat(e.timestamp)).seconds < 3600]
        
        errors_by_category = {}
        for error in self.error_log:
            category = error.category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
        
        errors_by_severity = {}
        for error in self.error_log:
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        recovery_attempts = [e for e in self.error_log if e.recovery_attempted]
        successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
        
        overall_recovery_rate = (len(successful_recoveries) / len(recovery_attempts) 
                               if recovery_attempts else 0)
        
        # Assess overall health
        health_status = 'good'
        if recent_errors:
            if len(recent_errors) > 5:
                health_status = 'poor'
            elif any(e.severity == ErrorSeverity.CRITICAL for e in recent_errors):
                health_status = 'critical'
            elif any(e.severity == ErrorSeverity.HIGH for e in recent_errors):
                health_status = 'degraded'
        
        return {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'errors_by_category': errors_by_category,
            'errors_by_severity': errors_by_severity,
            'recovery_attempts': len(recovery_attempts),
            'successful_recoveries': len(successful_recoveries),
            'overall_recovery_rate': overall_recovery_rate,
            'health_status': health_status,
            'component_health': self.health_status
        }

    def generate_error_report(self, output_file: str):
        """Generate comprehensive error report."""
        summary = self.get_error_summary()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_errors': [
                {
                    'error_id': e.error_id,
                    'timestamp': e.timestamp,
                    'category': e.category.value,
                    'severity': e.severity.value,
                    'description': e.description,
                    'recovery_attempted': e.recovery_attempted,
                    'recovery_successful': e.recovery_successful,
                    'recovery_actions': e.recovery_actions
                }
                for e in self.error_log
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error report saved to {output_file}")

    def clear_error_log(self):
        """Clear the error log (for testing or reset)."""
        self.error_log.clear()
        self.health_status.clear()
        logger.info("Error log cleared")


class ErrorRecoveryManager:
    """Manager for coordinating error recovery across the entire pipeline."""

    def __init__(self, pipeline_config: Dict[str, Any]):
        """Initialize error recovery manager.
        
        Args:
            pipeline_config: Pipeline configuration
        """
        self.config = pipeline_config
        self.error_handler = ErrorHandler(pipeline_config.get('output_dir', 'pipeline_output'))
        self.recovery_state = {}

    def safe_execute(self, 
                    func: Callable, 
                    *args, 
                    context: Dict[str, Any] = None,
                    retry_count: int = 3,
                    **kwargs) -> Tuple[Any, Optional[ErrorRecord]]:
        """Safely execute a function with error handling and recovery.
        
        Args:
            func: Function to execute
            args: Positional arguments for the function
            context: Context information for error handling
            retry_count: Number of retry attempts
            kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (result, error_record)
        """
        context = context or {}
        context['function'] = func.__name__
        
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                result = func(*args, **kwargs)
                return result, None
            
            except Exception as e:
                context['attempt'] = attempt + 1
                error_record = self.error_handler.handle_error(e, context)
                last_error = error_record
                
                # If recovery was successful, try again
                if error_record.recovery_successful and attempt < retry_count:
                    logger.info(f"Retrying after successful recovery (attempt {attempt + 2})")
                    continue
                
                # If this is the last attempt, break
                if attempt == retry_count:
                    break
        
        return None, last_error

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return self.error_handler.get_error_summary()

    def save_recovery_report(self, output_file: str):
        """Save comprehensive recovery report."""
        self.error_handler.generate_error_report(output_file)


# Context manager for error handling
class ErrorContext:
    """Context manager for automatic error handling."""

    def __init__(self, error_handler: ErrorHandler, context: Dict[str, Any]):
        """Initialize error context.
        
        Args:
            error_handler: Error handler instance
            context: Context information
        """
        self.error_handler = error_handler
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_error(exc_val, self.context)
        return False  # Don't suppress the exception


def main():
    """Main function for testing error handling."""
    # Example usage
    error_handler = ErrorHandler("test_workspace")
    
    # Simulate some errors
    try:
        raise FileNotFoundError("Test file not found")
    except Exception as e:
        error_handler.handle_error(e, {'phase': 'conversion', 'component': 'docling'})
    
    try:
        raise ValueError("Invalid configuration value")
    except Exception as e:
        error_handler.handle_error(e, {'phase': 'config', 'file': 'config.yaml'})
    
    # Generate report
    summary = error_handler.get_error_summary()
    print(json.dumps(summary, indent=2))
    
    error_handler.generate_error_report("error_report.json")
    print("Error report saved to error_report.json")


if __name__ == "__main__":
    main()