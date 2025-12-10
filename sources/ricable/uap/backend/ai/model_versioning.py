# File: backend/ai/model_versioning.py
# Model Versioning System
# Provides comprehensive model version control, comparison, and rollback capabilities

import asyncio
import json
import logging
import os
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import semantic_version

# Import existing infrastructure
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_model_version_event

# Configure logging
logger = logging.getLogger(__name__)

class VersionStatus(Enum):
    """Model version status states"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class VersionType(Enum):
    """Model version types based on semantic versioning"""
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes
    PRERELEASE = "prerelease"  # Alpha, beta, rc

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_id: str
    version: str
    semantic_version: str
    version_type: VersionType
    status: VersionStatus
    model_path: str
    model_hash: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    created_by: str
    parent_version: Optional[str] = None
    changelog: List[str] = None
    performance_metrics: Dict[str, float] = None
    file_size_mb: float = 0.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.changelog is None:
            self.changelog = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['version_type'] = self.version_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class VersionComparison:
    """Comparison between two model versions"""
    version_a: str
    version_b: str
    model_id: str
    size_diff_mb: float
    performance_diff: Dict[str, float]
    compatibility_issues: List[str]
    recommendation: str
    compared_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['compared_at'] = self.compared_at.isoformat()
        return data

class ModelVersionManager:
    """
    Advanced Model Versioning System.
    Handles version creation, comparison, rollback, and lifecycle management.
    """
    
    def __init__(self, base_version_dir: str = None):
        """
        Initialize the version manager.
        
        Args:
            base_version_dir: Base directory for version storage
        """
        self.base_version_dir = Path(base_version_dir) if base_version_dir else Path.home() / ".uap" / "versions"
        
        # Version tracking
        self.versions: Dict[str, ModelVersion] = {}  # version_id -> ModelVersion
        self.model_versions: Dict[str, List[str]] = {}  # model_id -> [version_ids]
        self.version_history: List[Dict[str, Any]] = []
        
        # Initialize directories
        self.base_version_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.base_version_dir / "models"
        self.metadata_dir = self.base_version_dir / "metadata"
        self.snapshots_dir = self.base_version_dir / "snapshots"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.snapshots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Version manager initialized with base directory: {self.base_version_dir}")
    
    async def initialize(self) -> bool:
        """
        Initialize the version manager and load existing versions.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing model version manager...")
            
            # Load existing versions
            await self._load_existing_versions()
            
            logger.info(f"Version manager initialization complete. Loaded {len(self.versions)} versions.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize version manager: {e}")
            return False
    
    async def create_version(self,
                           model_id: str,
                           model_path: str,
                           version_type: VersionType,
                           config: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None,
                           changelog: List[str] = None,
                           created_by: str = "system",
                           tags: List[str] = None) -> str:
        """
        Create a new model version.
        
        Args:
            model_id: Model identifier
            model_path: Path to model files
            version_type: Type of version (major, minor, patch)
            config: Model configuration
            metadata: Additional metadata
            changelog: List of changes
            created_by: Version creator
            tags: Version tags
            
        Returns:
            Version ID
        """
        try:
            # Generate version number
            version_number = await self._generate_version_number(model_id, version_type)
            version_id = f"{model_id}_{version_number}_{int(datetime.utcnow().timestamp())}"
            
            # Calculate model hash
            model_hash = await self._calculate_model_hash(model_path)
            
            # Get file size
            file_size_mb = await self._get_model_size(model_path)
            
            # Get parent version
            parent_version = await self._get_latest_version(model_id)
            
            # Create version record
            version = ModelVersion(
                version_id=version_id,
                model_id=model_id,
                version=version_number,
                semantic_version=version_number,
                version_type=version_type,
                status=VersionStatus.DEVELOPMENT,
                model_path=model_path,
                model_hash=model_hash,
                config=config or {},
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                created_by=created_by,
                parent_version=parent_version.version if parent_version else None,
                changelog=changelog or [],
                file_size_mb=file_size_mb,
                tags=tags or []
            )
            
            # Store version
            self.versions[version_id] = version
            
            # Update model version list
            if model_id not in self.model_versions:
                self.model_versions[model_id] = []
            self.model_versions[model_id].append(version_id)
            self.model_versions[model_id].sort(key=lambda vid: self.versions[vid].created_at)
            
            # Create version snapshot
            await self._create_version_snapshot(version)
            
            # Save metadata
            await self._save_version_metadata(version)
            
            # Log version creation
            uap_logger.log_event(
                LogLevel.INFO,
                f"Model version created: {model_id} v{version_number}",
                EventType.AGENT,
                {
                    "version_id": version_id,
                    "model_id": model_id,
                    "version": version_number,
                    "version_type": version_type.value,
                    "file_size_mb": file_size_mb
                },
                "version_manager"
            )
            
            # Record metrics
            record_model_version_event(model_id, version_number, "created")
            
            # Add to history
            self.version_history.append({
                "action": "create",
                "version_id": version_id,
                "model_id": model_id,
                "version": version_number,
                "timestamp": datetime.utcnow()
            })
            
            logger.info(f"Created version {version_number} for model {model_id} (ID: {version_id})")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create version for model {model_id}: {e}")
            raise
    
    async def _generate_version_number(self, model_id: str, version_type: VersionType) -> str:
        """
        Generate semantic version number.
        
        Args:
            model_id: Model identifier
            version_type: Version type
            
        Returns:
            Semantic version string
        """
        try:
            # Get latest version
            latest_version = await self._get_latest_version(model_id)
            
            if not latest_version:
                return "1.0.0"
            
            # Parse current version
            current_version = semantic_version.Version(latest_version.semantic_version)
            
            # Increment based on type
            if version_type == VersionType.MAJOR:
                new_version = current_version.next_major()
            elif version_type == VersionType.MINOR:
                new_version = current_version.next_minor()
            elif version_type == VersionType.PATCH:
                new_version = current_version.next_patch()
            else:  # PRERELEASE
                new_version = semantic_version.Version(f"{current_version.major}.{current_version.minor}.{current_version.patch}-alpha.1")
            
            return str(new_version)
            
        except Exception as e:
            logger.error(f"Failed to generate version number: {e}")
            return "1.0.0"
    
    async def _get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get the latest version of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Latest ModelVersion or None
        """
        if model_id not in self.model_versions:
            return None
        
        version_ids = self.model_versions[model_id]
        if not version_ids:
            return None
        
        # Get the most recent version
        latest_version_id = max(version_ids, key=lambda vid: self.versions[vid].created_at)
        return self.versions[latest_version_id]
    
    async def _calculate_model_hash(self, model_path: str) -> str:
        """
        Calculate hash of model files.
        
        Args:
            model_path: Path to model files
            
        Returns:
            SHA-256 hash
        """
        try:
            model_path_obj = Path(model_path)
            hasher = hashlib.sha256()
            
            if model_path_obj.is_file():
                # Single file
                with open(model_path_obj, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            elif model_path_obj.is_dir():
                # Directory - hash all files
                for file_path in sorted(model_path_obj.rglob('*')):
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate model hash: {e}")
            return "unknown"
    
    async def _get_model_size(self, model_path: str) -> float:
        """
        Get total size of model files in MB.
        
        Args:
            model_path: Path to model files
            
        Returns:
            Size in MB
        """
        try:
            model_path_obj = Path(model_path)
            total_size = 0
            
            if model_path_obj.is_file():
                total_size = model_path_obj.stat().st_size
            elif model_path_obj.is_dir():
                for file_path in model_path_obj.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Failed to get model size: {e}")
            return 0.0
    
    async def _create_version_snapshot(self, version: ModelVersion) -> None:
        """
        Create a snapshot of the model version.
        
        Args:
            version: Model version to snapshot
        """
        try:
            snapshot_dir = self.snapshots_dir / version.version_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            source_path = Path(version.model_path)
            
            if source_path.is_file():
                # Copy single file
                shutil.copy2(source_path, snapshot_dir / source_path.name)
            elif source_path.is_dir():
                # Copy entire directory
                target_dir = snapshot_dir / source_path.name
                shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
            
            # Update version path to snapshot
            version.model_path = str(snapshot_dir)
            
            logger.info(f"Created snapshot for version {version.version_id}")
            
        except Exception as e:
            logger.error(f"Failed to create version snapshot: {e}")
    
    async def promote_version(self, version_id: str, target_status: VersionStatus) -> bool:
        """
        Promote a version to a higher status.
        
        Args:
            version_id: Version to promote
            target_status: Target status
            
        Returns:
            True if promotion successful
        """
        try:
            if version_id not in self.versions:
                raise ValueError(f"Version {version_id} not found")
            
            version = self.versions[version_id]
            old_status = version.status
            
            # Validate promotion path
            valid_promotions = {
                VersionStatus.DEVELOPMENT: [VersionStatus.TESTING],
                VersionStatus.TESTING: [VersionStatus.STAGING],
                VersionStatus.STAGING: [VersionStatus.PRODUCTION],
                VersionStatus.PRODUCTION: [VersionStatus.DEPRECATED],
                VersionStatus.DEPRECATED: [VersionStatus.ARCHIVED]
            }
            
            if target_status not in valid_promotions.get(old_status, []):
                raise ValueError(f"Invalid promotion from {old_status.value} to {target_status.value}")
            
            # Update status
            version.status = target_status
            
            # Save metadata
            await self._save_version_metadata(version)
            
            # Log promotion
            uap_logger.log_event(
                LogLevel.INFO,
                f"Version promoted: {version.model_id} v{version.version} from {old_status.value} to {target_status.value}",
                EventType.AGENT,
                {
                    "version_id": version_id,
                    "model_id": version.model_id,
                    "version": version.version,
                    "old_status": old_status.value,
                    "new_status": target_status.value
                },
                "version_manager"
            )
            
            # Record metrics
            record_model_version_event(version.model_id, version.version, f"promoted_to_{target_status.value}")
            
            logger.info(f"Promoted version {version_id} from {old_status.value} to {target_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote version {version_id}: {e}")
            return False
    
    async def compare_versions(self, version_a_id: str, version_b_id: str) -> VersionComparison:
        """
        Compare two model versions.
        
        Args:
            version_a_id: First version ID
            version_b_id: Second version ID
            
        Returns:
            Version comparison result
        """
        try:
            if version_a_id not in self.versions or version_b_id not in self.versions:
                raise ValueError("One or both versions not found")
            
            version_a = self.versions[version_a_id]
            version_b = self.versions[version_b_id]
            
            if version_a.model_id != version_b.model_id:
                raise ValueError("Cannot compare versions from different models")
            
            # Calculate size difference
            size_diff = version_b.file_size_mb - version_a.file_size_mb
            
            # Compare performance metrics
            performance_diff = {}
            for metric in set(version_a.performance_metrics.keys()) | set(version_b.performance_metrics.keys()):
                val_a = version_a.performance_metrics.get(metric, 0.0)
                val_b = version_b.performance_metrics.get(metric, 0.0)
                performance_diff[metric] = val_b - val_a
            
            # Check compatibility issues
            compatibility_issues = []
            
            # Compare configs for breaking changes
            if version_a.config != version_b.config:
                compatibility_issues.append("Configuration changes detected")
            
            # Compare model hashes
            if version_a.model_hash != version_b.model_hash:
                compatibility_issues.append("Model weights have changed")
            
            # Generate recommendation
            recommendation = await self._generate_version_recommendation(
                version_a, version_b, performance_diff, compatibility_issues
            )
            
            comparison = VersionComparison(
                version_a=version_a.version,
                version_b=version_b.version,
                model_id=version_a.model_id,
                size_diff_mb=size_diff,
                performance_diff=performance_diff,
                compatibility_issues=compatibility_issues,
                recommendation=recommendation,
                compared_at=datetime.utcnow()
            )
            
            logger.info(f"Compared versions {version_a.version} and {version_b.version} for model {version_a.model_id}")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    async def _generate_version_recommendation(self,
                                             version_a: ModelVersion,
                                             version_b: ModelVersion,
                                             performance_diff: Dict[str, float],
                                             compatibility_issues: List[str]) -> str:
        """
        Generate recommendation based on version comparison.
        
        Args:
            version_a: First version
            version_b: Second version
            performance_diff: Performance differences
            compatibility_issues: Compatibility issues
            
        Returns:
            Recommendation string
        """
        try:
            # Parse version numbers for comparison
            ver_a = semantic_version.Version(version_a.semantic_version)
            ver_b = semantic_version.Version(version_b.semantic_version)
            
            if ver_b > ver_a:
                # Newer version
                if compatibility_issues:
                    return f"Upgrade to v{version_b.version} with caution - compatibility issues detected"
                elif any(diff > 0 for diff in performance_diff.values()):
                    return f"Recommended upgrade to v{version_b.version} - performance improvements detected"
                else:
                    return f"Consider upgrade to v{version_b.version} - newer version with potential improvements"
            elif ver_a > ver_b:
                # Older version
                return f"Rollback to v{version_b.version} not recommended unless critical issues exist"
            else:
                # Same version
                return "Versions are identical"
                
        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return "Unable to generate recommendation"
    
    async def rollback_to_version(self, model_id: str, target_version: str) -> bool:
        """
        Rollback model to a specific version.
        
        Args:
            model_id: Model identifier
            target_version: Target version to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            # Find target version
            target_version_obj = None
            for version_id, version in self.versions.items():
                if version.model_id == model_id and version.version == target_version:
                    target_version_obj = version
                    break
            
            if not target_version_obj:
                raise ValueError(f"Version {target_version} not found for model {model_id}")
            
            # Validate rollback (only allow rollback to production or staging versions)
            if target_version_obj.status not in [VersionStatus.PRODUCTION, VersionStatus.STAGING]:
                raise ValueError(f"Cannot rollback to version with status {target_version_obj.status.value}")
            
            # Log rollback
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Model rollback initiated: {model_id} to v{target_version}",
                EventType.AGENT,
                {
                    "model_id": model_id,
                    "target_version": target_version,
                    "version_id": target_version_obj.version_id
                },
                "version_manager"
            )
            
            # Record metrics
            record_model_version_event(model_id, target_version, "rollback")
            
            # Add to history
            self.version_history.append({
                "action": "rollback",
                "version_id": target_version_obj.version_id,
                "model_id": model_id,
                "version": target_version,
                "timestamp": datetime.utcnow()
            })
            
            logger.info(f"Rollback completed for model {model_id} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model {model_id} to version {target_version}: {e}")
            return False
    
    async def list_versions(self, 
                           model_id: str = None, 
                           status: VersionStatus = None,
                           limit: int = None) -> List[Dict[str, Any]]:
        """
        List model versions with optional filtering.
        
        Args:
            model_id: Filter by model ID
            status: Filter by version status
            limit: Maximum number of versions to return
            
        Returns:
            List of version information
        """
        versions = []
        
        for version_id, version in self.versions.items():
            if model_id and version.model_id != model_id:
                continue
            
            if status and version.status != status:
                continue
            
            versions.append(version.to_dict())
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v['created_at'], reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    async def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific version.
        
        Args:
            version_id: Version ID
            
        Returns:
            Version information or None
        """
        if version_id not in self.versions:
            return None
        
        version = self.versions[version_id]
        version_info = version.to_dict()
        
        # Add additional information
        version_info["is_latest"] = await self._is_latest_version(version)
        version_info["snapshot_exists"] = (self.snapshots_dir / version_id).exists()
        
        return version_info
    
    async def _is_latest_version(self, version: ModelVersion) -> bool:
        """
        Check if a version is the latest for its model.
        
        Args:
            version: Version to check
            
        Returns:
            True if latest version
        """
        latest_version = await self._get_latest_version(version.model_id)
        return latest_version and latest_version.version_id == version.version_id
    
    async def _load_existing_versions(self) -> None:
        """
        Load existing versions from metadata files.
        """
        try:
            metadata_files = list(self.metadata_dir.glob("version_*.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        version_data = json.load(f)
                    
                    # Reconstruct version object
                    version = ModelVersion(
                        version_id=version_data["version_id"],
                        model_id=version_data["model_id"],
                        version=version_data["version"],
                        semantic_version=version_data["semantic_version"],
                        version_type=VersionType(version_data["version_type"]),
                        status=VersionStatus(version_data["status"]),
                        model_path=version_data["model_path"],
                        model_hash=version_data["model_hash"],
                        config=version_data["config"],
                        metadata=version_data["metadata"],
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        created_by=version_data["created_by"],
                        parent_version=version_data.get("parent_version"),
                        changelog=version_data.get("changelog", []),
                        performance_metrics=version_data.get("performance_metrics", {}),
                        file_size_mb=version_data.get("file_size_mb", 0.0),
                        tags=version_data.get("tags", [])
                    )
                    
                    self.versions[version.version_id] = version
                    
                    # Update model version list
                    if version.model_id not in self.model_versions:
                        self.model_versions[version.model_id] = []
                    self.model_versions[version.model_id].append(version.version_id)
                    
                except Exception as e:
                    logger.error(f"Failed to load version metadata from {metadata_file}: {e}")
            
            # Sort version lists
            for model_id in self.model_versions:
                self.model_versions[model_id].sort(key=lambda vid: self.versions[vid].created_at)
            
            logger.info(f"Loaded {len(self.versions)} existing versions")
            
        except Exception as e:
            logger.error(f"Failed to load existing versions: {e}")
    
    async def _save_version_metadata(self, version: ModelVersion) -> None:
        """
        Save version metadata to file.
        
        Args:
            version: Version to save
        """
        try:
            metadata_file = self.metadata_dir / f"version_{version.version_id}.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")
    
    async def get_manager_status(self) -> Dict[str, Any]:
        """
        Get version manager status.
        
        Returns:
            Manager status information
        """
        total_versions = len(self.versions)
        models_with_versions = len(self.model_versions)
        
        # Count versions by status
        status_counts = {}
        for status in VersionStatus:
            status_counts[status.value] = len([v for v in self.versions.values() if v.status == status])
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_versions": total_versions,
            "models_with_versions": models_with_versions,
            "status_distribution": status_counts,
            "recent_activity": self.version_history[-10:] if self.version_history else [],
            "base_version_dir": str(self.base_version_dir)
        }

# Global version manager instance
_version_manager = None

def get_version_manager() -> ModelVersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager()
    return _version_manager

print("Model Versioning System loaded.")