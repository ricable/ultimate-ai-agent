"""
Projects Services Package

This package contains all services related to project management,
including project CRUD operations, task management, document management,
versioning, progress tracking, source linking, and AI-assisted project creation.
"""

from .document_service import DocumentService
from .progress_service import ProgressService, progress_service
from .project_creation_service import ProjectCreationService
from .project_service import ProjectService
from .source_linking_service import SourceLinkingService
from .task_service import TaskService
from .versioning_service import VersioningService

__all__ = [
    "ProjectService",
    "TaskService",
    "DocumentService",
    "VersioningService",
    "ProgressService",
    "progress_service",
    "ProjectCreationService",
    "SourceLinkingService",
]
