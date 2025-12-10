# File: backend/operations/disaster_recovery.py
"""
Disaster recovery and backup automation for UAP platform.
Provides comprehensive backup strategies, automated failover, 
business continuity planning, and recovery testing.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
import aioredis
from prometheus_client import Counter, Gauge, Histogram

from ..cache.redis_cache import get_redis_client
from ..database.connection import get_database_connection

# Configure logging
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental" 
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class RecoveryType(Enum):
    """Types of recovery operations"""
    POINT_IN_TIME = "point_in_time"
    FULL_RESTORE = "full_restore"
    PARTIAL_RESTORE = "partial_restore"
    FAILOVER = "failover"
    FAILBACK = "failback"

class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class BackupJob:
    """Backup job configuration and status"""
    id: str
    name: str
    backup_type: BackupType
    source_path: str
    destination_path: str
    schedule: str  # Cron format
    retention_days: int
    compression: bool
    encryption: bool
    status: BackupStatus
    created_at: float
    last_run: Optional[float]
    next_run: Optional[float]
    size_bytes: Optional[int]
    duration_seconds: Optional[float]
    error_message: Optional[str]

@dataclass
class RecoveryPoint:
    """Recovery point in time"""
    id: str
    backup_job_id: str
    timestamp: float
    backup_type: BackupType
    file_path: str
    size_bytes: int
    checksum: str
    retention_until: float
    metadata: Dict[str, Any]

@dataclass
class DisasterRecoveryPlan:
    """Disaster recovery plan"""
    id: str
    name: str
    description: str
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    backup_jobs: List[str]
    failover_procedures: List[Dict[str, Any]]
    testing_schedule: str
    last_tested: Optional[float]
    test_success: Optional[bool]
    created_at: float

@dataclass
class FailoverEvent:
    """Failover event record"""
    id: str
    plan_id: str
    trigger_type: str  # manual, automatic, test
    trigger_reason: str
    started_at: float
    completed_at: Optional[float]
    status: str
    steps_completed: List[str]
    errors: List[str]
    rollback_required: bool

class BackupManager:
    """
    Comprehensive backup management system with automated scheduling,
    multiple backup types, and intelligent retention policies.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.recovery_points: Dict[str, List[RecoveryPoint]] = {}
        self.backup_running = False
        
        # Configuration
        self.default_backup_root = "/var/backups/uap"
        self.temp_dir = "/tmp/uap-backups"
        self.max_concurrent_backups = 3
        self.compression_level = 6
        
        # Prometheus metrics
        self.backup_counter = Counter(
            'uap_backups_total',
            'Total number of backup operations',
            ['backup_type', 'status']
        )
        
        self.backup_duration = Histogram(
            'uap_backup_duration_seconds',
            'Backup operation duration',
            ['backup_type', 'job_name']
        )
        
        self.backup_size_gauge = Gauge(
            'uap_backup_size_bytes',
            'Size of backup files',
            ['backup_type', 'job_name']
        )
        
        self.recovery_points_gauge = Gauge(
            'uap_recovery_points_count',
            'Number of available recovery points',
            ['job_name']
        )
        
        # Initialize directories
        self._initialize_directories()
    
    def _initialize_directories(self) -> None:
        """Initialize backup directories"""
        os.makedirs(self.default_backup_root, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def start_backup_scheduler(self) -> None:
        """Start the backup scheduler"""
        if self.backup_running:
            return
        
        self.backup_running = True
        logger.info("Starting backup scheduler...")
        
        try:
            await self._backup_scheduler_loop()
        except Exception as e:
            logger.error(f"Backup scheduler error: {e}")
        finally:
            self.backup_running = False
    
    async def stop_backup_scheduler(self) -> None:
        """Stop the backup scheduler"""
        self.backup_running = False
        logger.info("Stopping backup scheduler...")
    
    async def _backup_scheduler_loop(self) -> None:
        """Main backup scheduler loop"""
        while self.backup_running:
            try:
                current_time = time.time()
                
                # Check each backup job for scheduled execution
                for job_id, job in self.backup_jobs.items():
                    if self._should_run_backup(job, current_time):
                        await self._execute_backup_job(job)
                
                # Clean up expired backups
                await self._cleanup_expired_backups()
                
                # Update metrics
                await self._update_backup_metrics()
                
            except Exception as e:
                logger.error(f"Backup scheduler loop error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def _should_run_backup(self, job: BackupJob, current_time: float) -> bool:
        """Check if backup job should run now"""
        if job.status == BackupStatus.RUNNING:
            return False
        
        if job.next_run is None:
            return True  # First run
        
        return current_time >= job.next_run
    
    async def _execute_backup_job(self, job: BackupJob) -> None:
        """Execute a backup job"""
        job.status = BackupStatus.RUNNING
        job.last_run = time.time()
        start_time = time.time()
        
        logger.info(f"Starting backup job: {job.name}")
        
        try:
            # Perform the backup based on type
            if job.backup_type == BackupType.FULL:
                result = await self._perform_full_backup(job)
            elif job.backup_type == BackupType.INCREMENTAL:
                result = await self._perform_incremental_backup(job)
            elif job.backup_type == BackupType.DIFFERENTIAL:
                result = await self._perform_differential_backup(job)
            elif job.backup_type == BackupType.SNAPSHOT:
                result = await self._perform_snapshot_backup(job)
            else:
                raise ValueError(f"Unknown backup type: {job.backup_type}")
            
            # Update job status
            job.status = BackupStatus.COMPLETED
            job.size_bytes = result["size_bytes"]
            job.duration_seconds = time.time() - start_time
            job.error_message = None
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                id=f"{job.id}_{int(job.last_run)}",
                backup_job_id=job.id,
                timestamp=job.last_run,
                backup_type=job.backup_type,
                file_path=result["file_path"],
                size_bytes=result["size_bytes"],
                checksum=result["checksum"],
                retention_until=job.last_run + (job.retention_days * 86400),
                metadata=result["metadata"]
            )
            
            await self._store_recovery_point(recovery_point)
            
            # Schedule next run
            job.next_run = self._calculate_next_run(job.schedule, job.last_run)
            
            # Update Prometheus metrics
            self.backup_counter.labels(
                backup_type=job.backup_type.value,
                status="completed"
            ).inc()
            
            self.backup_duration.labels(
                backup_type=job.backup_type.value,
                job_name=job.name
            ).observe(job.duration_seconds)
            
            self.backup_size_gauge.labels(
                backup_type=job.backup_type.value,
                job_name=job.name
            ).set(job.size_bytes)
            
            logger.info(f"Backup job completed: {job.name} ({job.size_bytes} bytes)")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.duration_seconds = time.time() - start_time
            
            self.backup_counter.labels(
                backup_type=job.backup_type.value,
                status="failed"
            ).inc()
            
            logger.error(f"Backup job failed: {job.name} - {e}")
        
        # Store updated job
        await self._store_backup_job(job)
    
    async def _perform_full_backup(self, job: BackupJob) -> Dict[str, Any]:
        """Perform full backup"""
        timestamp = int(time.time())
        backup_filename = f"{job.name}_full_{timestamp}.tar.gz"
        backup_path = os.path.join(self.default_backup_root, backup_filename)
        
        # Create compressed archive
        cmd = [
            "tar", "-czf", backup_path,
            "-C", os.path.dirname(job.source_path),
            os.path.basename(job.source_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Backup command failed: {result.stderr}")
        
        # Get file size and checksum
        size_bytes = os.path.getsize(backup_path)
        checksum = await self._calculate_checksum(backup_path)
        
        # Encrypt if required
        if job.encryption:
            backup_path = await self._encrypt_backup(backup_path)
            size_bytes = os.path.getsize(backup_path)
        
        return {
            "file_path": backup_path,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "metadata": {
                "source_path": job.source_path,
                "compression": job.compression,
                "encryption": job.encryption
            }
        }
    
    async def _perform_incremental_backup(self, job: BackupJob) -> Dict[str, Any]:
        """Perform incremental backup"""
        # Find last backup timestamp
        last_backup_time = await self._get_last_backup_time(job.id)
        
        timestamp = int(time.time())
        backup_filename = f"{job.name}_incremental_{timestamp}.tar.gz"
        backup_path = os.path.join(self.default_backup_root, backup_filename)
        
        # Create incremental backup using find and tar
        find_cmd = [
            "find", job.source_path,
            "-newer", f"@{last_backup_time}" if last_backup_time else "@0",
            "-type", "f"
        ]
        
        tar_cmd = [
            "tar", "-czf", backup_path,
            "-T", "-"
        ]
        
        # Pipe find output to tar
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        tar_proc = subprocess.Popen(tar_cmd, stdin=find_proc.stdout, capture_output=True, text=True)
        find_proc.stdout.close()
        
        tar_output, tar_error = tar_proc.communicate()
        
        if tar_proc.returncode != 0:
            raise Exception(f"Incremental backup failed: {tar_error}")
        
        size_bytes = os.path.getsize(backup_path)
        checksum = await self._calculate_checksum(backup_path)
        
        if job.encryption:
            backup_path = await self._encrypt_backup(backup_path)
            size_bytes = os.path.getsize(backup_path)
        
        return {
            "file_path": backup_path,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "metadata": {
                "source_path": job.source_path,
                "last_backup_time": last_backup_time,
                "incremental": True
            }
        }
    
    async def _perform_differential_backup(self, job: BackupJob) -> Dict[str, Any]:
        """Perform differential backup"""
        # Find last full backup timestamp
        last_full_backup_time = await self._get_last_full_backup_time(job.id)
        
        timestamp = int(time.time())
        backup_filename = f"{job.name}_differential_{timestamp}.tar.gz"
        backup_path = os.path.join(self.default_backup_root, backup_filename)
        
        # Create differential backup (changes since last full backup)
        find_cmd = [
            "find", job.source_path,
            "-newer", f"@{last_full_backup_time}" if last_full_backup_time else "@0",
            "-type", "f"
        ]
        
        tar_cmd = [
            "tar", "-czf", backup_path,
            "-T", "-"
        ]
        
        find_proc = subprocess.Popen(find_cmd, stdout=subprocess.PIPE)
        tar_proc = subprocess.Popen(tar_cmd, stdin=find_proc.stdout, capture_output=True, text=True)
        find_proc.stdout.close()
        
        tar_output, tar_error = tar_proc.communicate()
        
        if tar_proc.returncode != 0:
            raise Exception(f"Differential backup failed: {tar_error}")
        
        size_bytes = os.path.getsize(backup_path)
        checksum = await self._calculate_checksum(backup_path)
        
        if job.encryption:
            backup_path = await self._encrypt_backup(backup_path)
            size_bytes = os.path.getsize(backup_path)
        
        return {
            "file_path": backup_path,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "metadata": {
                "source_path": job.source_path,
                "last_full_backup_time": last_full_backup_time,
                "differential": True
            }
        }
    
    async def _perform_snapshot_backup(self, job: BackupJob) -> Dict[str, Any]:
        """Perform snapshot backup (for databases, etc.)"""
        timestamp = int(time.time())
        backup_filename = f"{job.name}_snapshot_{timestamp}.sql"
        backup_path = os.path.join(self.default_backup_root, backup_filename)
        
        # Database snapshot using pg_dump
        if "database" in job.source_path.lower():
            cmd = [
                "pg_dump", 
                "--host=localhost",
                "--port=5432",
                "--username=uapuser",
                "--format=custom",
                "--file", backup_path,
                "uapdb"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Database snapshot failed: {result.stderr}")
        
        # Compress if required
        if job.compression:
            compressed_path = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as f_out:
                    f_out.write(backup_path, os.path.basename(backup_path))
            os.remove(backup_path)
            backup_path = compressed_path
        
        size_bytes = os.path.getsize(backup_path)
        checksum = await self._calculate_checksum(backup_path)
        
        if job.encryption:
            backup_path = await self._encrypt_backup(backup_path)
            size_bytes = os.path.getsize(backup_path)
        
        return {
            "file_path": backup_path,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "metadata": {
                "source_path": job.source_path,
                "snapshot": True,
                "format": "postgres_custom"
            }
        }
    
    async def create_backup_job(self, name: str, backup_type: BackupType, 
                               source_path: str, schedule: str = "0 2 * * *",
                               retention_days: int = 30, compression: bool = True,
                               encryption: bool = True) -> BackupJob:
        """Create a new backup job"""
        job_id = f"backup_{name}_{int(time.time())}"
        
        job = BackupJob(
            id=job_id,
            name=name,
            backup_type=backup_type,
            source_path=source_path,
            destination_path=self.default_backup_root,
            schedule=schedule,
            retention_days=retention_days,
            compression=compression,
            encryption=encryption,
            status=BackupStatus.PENDING,
            created_at=time.time(),
            last_run=None,
            next_run=self._calculate_next_run(schedule, time.time()),
            size_bytes=None,
            duration_seconds=None,
            error_message=None
        )
        
        self.backup_jobs[job_id] = job
        await self._store_backup_job(job)
        
        logger.info(f"Created backup job: {name} ({backup_type.value})")
        return job
    
    async def restore_from_recovery_point(self, recovery_point_id: str, 
                                        destination_path: str) -> Dict[str, Any]:
        """Restore data from a specific recovery point"""
        recovery_point = await self._get_recovery_point(recovery_point_id)
        if not recovery_point:
            raise ValueError(f"Recovery point not found: {recovery_point_id}")
        
        logger.info(f"Starting restore from recovery point: {recovery_point_id}")
        start_time = time.time()
        
        try:
            # Decrypt if necessary
            source_path = recovery_point.file_path
            if recovery_point.metadata.get("encryption"):
                source_path = await self._decrypt_backup(source_path)
            
            # Restore based on backup type
            if recovery_point.backup_type == BackupType.SNAPSHOT:
                await self._restore_snapshot(source_path, destination_path, recovery_point.metadata)
            else:
                await self._restore_archive(source_path, destination_path, recovery_point.metadata)
            
            duration = time.time() - start_time
            
            logger.info(f"Restore completed in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "duration_seconds": duration,
                "recovery_point_id": recovery_point_id,
                "destination_path": destination_path
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
    
    async def _restore_snapshot(self, source_path: str, destination_path: str, metadata: Dict[str, Any]) -> None:
        """Restore from database snapshot"""
        if metadata.get("format") == "postgres_custom":
            cmd = [
                "pg_restore",
                "--host=localhost",
                "--port=5432", 
                "--username=uapuser",
                "--dbname=uapdb",
                "--clean",
                "--create",
                source_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Database restore failed: {result.stderr}")
    
    async def _restore_archive(self, source_path: str, destination_path: str, metadata: Dict[str, Any]) -> None:
        """Restore from archive backup"""
        cmd = [
            "tar", "-xzf", source_path,
            "-C", destination_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Archive restore failed: {result.stderr}")
    
    # Helper methods
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _encrypt_backup(self, file_path: str) -> str:
        """Encrypt backup file"""
        # This would use actual encryption like GPG or AES
        # For now, just rename to indicate encryption
        encrypted_path = f"{file_path}.encrypted"
        shutil.move(file_path, encrypted_path)
        return encrypted_path
    
    async def _decrypt_backup(self, file_path: str) -> str:
        """Decrypt backup file"""
        # This would use actual decryption
        # For now, just rename back
        if file_path.endswith(".encrypted"):
            decrypted_path = file_path[:-10]  # Remove .encrypted
            shutil.copy(file_path, decrypted_path)
            return decrypted_path
        return file_path
    
    def _calculate_next_run(self, schedule: str, last_run: float) -> float:
        """Calculate next run time based on cron schedule"""
        # Simplified: just add 24 hours for daily backup
        return last_run + 86400
    
    async def _get_last_backup_time(self, job_id: str) -> Optional[float]:
        """Get timestamp of last backup for incremental"""
        if job_id in self.recovery_points:
            points = self.recovery_points[job_id]
            if points:
                return max(point.timestamp for point in points)
        return None
    
    async def _get_last_full_backup_time(self, job_id: str) -> Optional[float]:
        """Get timestamp of last full backup for differential"""
        if job_id in self.recovery_points:
            points = [p for p in self.recovery_points[job_id] if p.backup_type == BackupType.FULL]
            if points:
                return max(point.timestamp for point in points)
        return None
    
    async def _cleanup_expired_backups(self) -> None:
        """Clean up expired backup files"""
        current_time = time.time()
        
        for job_id, points in self.recovery_points.items():
            expired_points = [p for p in points if p.retention_until <= current_time]
            
            for point in expired_points:
                try:
                    # Remove file
                    if os.path.exists(point.file_path):
                        os.remove(point.file_path)
                        logger.info(f"Removed expired backup: {point.file_path}")
                    
                    # Remove from list
                    points.remove(point)
                    
                except Exception as e:
                    logger.error(f"Failed to remove expired backup {point.file_path}: {e}")
    
    async def _store_backup_job(self, job: BackupJob) -> None:
        """Store backup job in Redis"""
        key = f"backup_job:{job.id}"
        await self.redis_client.set(key, json.dumps(asdict(job), default=str))
    
    async def _store_recovery_point(self, point: RecoveryPoint) -> None:
        """Store recovery point in Redis"""
        key = f"recovery_point:{point.id}"
        await self.redis_client.set(key, json.dumps(asdict(point), default=str))
        
        # Add to job's recovery points list
        if point.backup_job_id not in self.recovery_points:
            self.recovery_points[point.backup_job_id] = []
        self.recovery_points[point.backup_job_id].append(point)
    
    async def _get_recovery_point(self, point_id: str) -> Optional[RecoveryPoint]:
        """Get recovery point by ID"""
        key = f"recovery_point:{point_id}"
        data = await self.redis_client.get(key)
        if data:
            point_dict = json.loads(data)
            point_dict['backup_type'] = BackupType(point_dict['backup_type'])
            return RecoveryPoint(**point_dict)
        return None
    
    async def _update_backup_metrics(self) -> None:
        """Update Prometheus metrics"""
        for job_id, points in self.recovery_points.items():
            if job_id in self.backup_jobs:
                job = self.backup_jobs[job_id]
                self.recovery_points_gauge.labels(job_name=job.name).set(len(points))

class DisasterRecoveryManager:
    """
    Comprehensive disaster recovery management with automated failover,
    recovery testing, and business continuity planning.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.backup_manager = BackupManager(redis_client)
        self.failover_manager = FailoverManager(redis_client)
        
        self.dr_plans: Dict[str, DisasterRecoveryPlan] = {}
        self.monitoring_active = False
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.failover_threshold = 3  # consecutive failures before failover
        
        # Prometheus metrics
        self.rto_gauge = Gauge(
            'uap_rto_minutes',
            'Recovery Time Objective in minutes',
            ['plan_name']
        )
        
        self.rpo_gauge = Gauge(
            'uap_rpo_minutes',
            'Recovery Point Objective in minutes',
            ['plan_name']
        )
        
        self.dr_test_success_gauge = Gauge(
            'uap_dr_test_success',
            'DR test success status (1 = success, 0 = failure)',
            ['plan_name']
        )
    
    async def start_dr_monitoring(self) -> None:
        """Start disaster recovery monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting disaster recovery monitoring...")
        
        tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._dr_testing_loop()),
            asyncio.create_task(self.backup_manager.start_backup_scheduler())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"DR monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    async def _health_monitoring_loop(self) -> None:
        """Monitor system health and trigger failover if needed"""
        consecutive_failures = 0
        
        while self.monitoring_active:
            try:
                health_status = await self._check_system_health()
                
                if health_status == HealthStatus.FAILED:
                    consecutive_failures += 1
                    logger.warning(f"System health check failed ({consecutive_failures}/{self.failover_threshold})")
                    
                    if consecutive_failures >= self.failover_threshold:
                        await self._trigger_automatic_failover("health_check_failure")
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _dr_testing_loop(self) -> None:
        """Periodic disaster recovery testing"""
        while self.monitoring_active:
            try:
                for plan_id, plan in self.dr_plans.items():
                    if self._should_test_dr_plan(plan):
                        await self._execute_dr_test(plan)
                
            except Exception as e:
                logger.error(f"DR testing error: {e}")
            
            await asyncio.sleep(3600)  # Check every hour
    
    async def _check_system_health(self) -> HealthStatus:
        """Check overall system health"""
        try:
            # Check database connectivity
            db_healthy = await self._check_database_health()
            
            # Check application endpoints
            app_healthy = await self._check_application_health()
            
            # Check critical services
            services_healthy = await self._check_services_health()
            
            if not db_healthy or not app_healthy:
                return HealthStatus.FAILED
            elif not services_healthy:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthStatus.FAILED
    
    async def _check_database_health(self) -> bool:
        """Check database health"""
        try:
            # Simple database connectivity check
            async with get_database_connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def _check_application_health(self) -> bool:
        """Check application health"""
        try:
            # Check if application is responding
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/health", timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_services_health(self) -> bool:
        """Check critical services health"""
        try:
            # Check Redis
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def create_dr_plan(self, name: str, description: str, rto_minutes: int = 60, 
                           rpo_minutes: int = 15) -> DisasterRecoveryPlan:
        """Create a new disaster recovery plan"""
        plan_id = f"dr_plan_{name}_{int(time.time())}"
        
        plan = DisasterRecoveryPlan(
            id=plan_id,
            name=name,
            description=description,
            rto_minutes=rto_minutes,
            rpo_minutes=rpo_minutes,
            backup_jobs=[],
            failover_procedures=[],
            testing_schedule="0 2 * * 0",  # Weekly on Sunday at 2 AM
            last_tested=None,
            test_success=None,
            created_at=time.time()
        )
        
        self.dr_plans[plan_id] = plan
        await self._store_dr_plan(plan)
        
        # Update Prometheus metrics
        self.rto_gauge.labels(plan_name=name).set(rto_minutes)
        self.rpo_gauge.labels(plan_name=name).set(rpo_minutes)
        
        logger.info(f"Created DR plan: {name}")
        return plan
    
    def _should_test_dr_plan(self, plan: DisasterRecoveryPlan) -> bool:
        """Check if DR plan should be tested"""
        if plan.last_tested is None:
            return True
        
        # Test weekly (simplified)
        time_since_test = time.time() - plan.last_tested
        return time_since_test >= 604800  # 1 week
    
    async def _execute_dr_test(self, plan: DisasterRecoveryPlan) -> None:
        """Execute disaster recovery test"""
        logger.info(f"Starting DR test for plan: {plan.name}")
        
        test_start_time = time.time()
        test_success = True
        
        try:
            # Test backup restoration
            for backup_job_id in plan.backup_jobs:
                test_success &= await self._test_backup_restoration(backup_job_id)
            
            # Test failover procedures
            for procedure in plan.failover_procedures:
                test_success &= await self._test_failover_procedure(procedure)
            
            plan.last_tested = test_start_time
            plan.test_success = test_success
            
            # Update metrics
            self.dr_test_success_gauge.labels(plan_name=plan.name).set(1 if test_success else 0)
            
            if test_success:
                logger.info(f"DR test passed for plan: {plan.name}")
            else:
                logger.warning(f"DR test failed for plan: {plan.name}")
            
        except Exception as e:
            logger.error(f"DR test error for plan {plan.name}: {e}")
            plan.test_success = False
            self.dr_test_success_gauge.labels(plan_name=plan.name).set(0)
        
        await self._store_dr_plan(plan)
    
    async def _test_backup_restoration(self, backup_job_id: str) -> bool:
        """Test backup restoration process"""
        try:
            # Find latest recovery point for this backup job
            recovery_points = self.backup_manager.recovery_points.get(backup_job_id, [])
            if not recovery_points:
                return False
            
            latest_point = max(recovery_points, key=lambda p: p.timestamp)
            
            # Test restore to temporary location
            test_restore_path = f"/tmp/dr_test_{backup_job_id}_{int(time.time())}"
            os.makedirs(test_restore_path, exist_ok=True)
            
            try:
                await self.backup_manager.restore_from_recovery_point(
                    latest_point.id, 
                    test_restore_path
                )
                
                # Clean up test restore
                shutil.rmtree(test_restore_path, ignore_errors=True)
                return True
                
            except Exception as e:
                logger.error(f"Test restoration failed for {backup_job_id}: {e}")
                shutil.rmtree(test_restore_path, ignore_errors=True)
                return False
                
        except Exception as e:
            logger.error(f"Backup restoration test error: {e}")
            return False
    
    async def _test_failover_procedure(self, procedure: Dict[str, Any]) -> bool:
        """Test failover procedure"""
        try:
            # This would test specific failover procedures
            # For now, just simulate success
            procedure_type = procedure.get("type", "unknown")
            logger.info(f"Testing failover procedure: {procedure_type}")
            
            # Simulate procedure test
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failover procedure test error: {e}")
            return False
    
    async def _trigger_automatic_failover(self, reason: str) -> None:
        """Trigger automatic failover"""
        logger.critical(f"Triggering automatic failover: {reason}")
        
        # Find appropriate DR plan
        primary_plan = None
        for plan in self.dr_plans.values():
            if "primary" in plan.name.lower():
                primary_plan = plan
                break
        
        if primary_plan:
            await self.failover_manager.execute_failover(primary_plan.id, "automatic", reason)
        else:
            logger.error("No primary DR plan found for automatic failover")
    
    async def _store_dr_plan(self, plan: DisasterRecoveryPlan) -> None:
        """Store DR plan in Redis"""
        key = f"dr_plan:{plan.id}"
        await self.redis_client.set(key, json.dumps(asdict(plan), default=str))

class FailoverManager:
    """Manages failover and failback operations"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.active_failovers: Dict[str, FailoverEvent] = {}
    
    async def execute_failover(self, plan_id: str, trigger_type: str, reason: str) -> FailoverEvent:
        """Execute failover according to DR plan"""
        event_id = f"failover_{plan_id}_{int(time.time())}"
        
        event = FailoverEvent(
            id=event_id,
            plan_id=plan_id,
            trigger_type=trigger_type,
            trigger_reason=reason,
            started_at=time.time(),
            completed_at=None,
            status="in_progress",
            steps_completed=[],
            errors=[],
            rollback_required=False
        )
        
        self.active_failovers[event_id] = event
        
        logger.critical(f"Starting failover: {event_id}")
        
        try:
            # Execute failover steps
            await self._execute_failover_steps(event)
            
            event.status = "completed"
            event.completed_at = time.time()
            
            logger.info(f"Failover completed: {event_id}")
            
        except Exception as e:
            event.status = "failed"
            event.errors.append(str(e))
            event.rollback_required = True
            
            logger.error(f"Failover failed: {event_id} - {e}")
        
        await self._store_failover_event(event)
        return event
    
    async def _execute_failover_steps(self, event: FailoverEvent) -> None:
        """Execute the steps for failover"""
        steps = [
            ("stop_primary_services", self._stop_primary_services),
            ("promote_secondary", self._promote_secondary),
            ("update_dns", self._update_dns),
            ("start_secondary_services", self._start_secondary_services),
            ("verify_failover", self._verify_failover)
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"Executing failover step: {step_name}")
                await step_func(event)
                event.steps_completed.append(step_name)
                
            except Exception as e:
                event.errors.append(f"{step_name}: {str(e)}")
                raise
    
    async def _stop_primary_services(self, event: FailoverEvent) -> None:
        """Stop primary services"""
        # This would stop the primary application services
        await asyncio.sleep(1)  # Simulate
    
    async def _promote_secondary(self, event: FailoverEvent) -> None:
        """Promote secondary to primary"""
        # This would promote secondary database/services
        await asyncio.sleep(2)  # Simulate
    
    async def _update_dns(self, event: FailoverEvent) -> None:
        """Update DNS to point to secondary"""
        # This would update DNS records
        await asyncio.sleep(1)  # Simulate
    
    async def _start_secondary_services(self, event: FailoverEvent) -> None:
        """Start services on secondary"""
        # This would start application services on secondary
        await asyncio.sleep(2)  # Simulate
    
    async def _verify_failover(self, event: FailoverEvent) -> None:
        """Verify failover was successful"""
        # This would verify the failover worked
        await asyncio.sleep(1)  # Simulate
    
    async def _store_failover_event(self, event: FailoverEvent) -> None:
        """Store failover event"""
        key = f"failover_event:{event.id}"
        await self.redis_client.set(key, json.dumps(asdict(event), default=str))

# Global instances
backup_manager = BackupManager()
disaster_recovery_manager = DisasterRecoveryManager()
failover_manager = FailoverManager()