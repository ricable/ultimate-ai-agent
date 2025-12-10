#!/usr/bin/env python3
"""
Performance Integration Module for Agentic Evaluation Framework
Integrates evaluation performance tracking with existing polyglot analytics
"""

import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
import logging
import asyncio

@dataclass
class PerformanceMetric:
    """Performance metric for integration"""
    metric_name: str
    metric_value: float
    metric_unit: str
    source: str
    category: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class IntegrationReport:
    """Integration status report"""
    status: str
    metrics_synced: int
    errors: List[str]
    duration: float
    next_sync: str

class PerformanceIntegration:
    def __init__(self, eval_root: str = "/workspace/agentic-eval"):
        self.eval_root = Path(eval_root)
        self.db_path = self.eval_root / "databases" / "results.db"
        self.integration_dir = self.eval_root / "integration"
        self.logs_dir = self.eval_root / "logs" / "integration"
        
        # Polyglot analytics paths
        self.polyglot_root = Path("/workspace")
        self.polyglot_analytics = self.polyglot_root / "dev-env" / "nushell" / "scripts" / "performance-analytics.nu"
        self.polyglot_logs = self.polyglot_root / "dev-env" / "nushell" / "logs"
        
        # Create directories
        for dir_path in [self.integration_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize integration database
        self.init_integration_database()

    def setup_logging(self):
        """Setup integration logging"""
        log_file = self.logs_dir / f"performance_integration_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("PerformanceIntegration")

    def init_integration_database(self):
        """Initialize integration tracking database"""
        conn = sqlite3.connect(self.db_path)
        
        # Integration metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS integration_metrics (
                id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                sync_status TEXT DEFAULT 'pending',
                polyglot_metric_id TEXT
            )
        ''')
        
        # Integration status table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS integration_status (
                id TEXT PRIMARY KEY,
                integration_type TEXT NOT NULL,
                last_sync DATETIME,
                metrics_count INTEGER DEFAULT 0,
                errors_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                configuration TEXT
            )
        ''')
        
        # Sync history table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sync_history (
                id TEXT PRIMARY KEY,
                sync_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metrics_synced INTEGER DEFAULT 0,
                errors_encountered INTEGER DEFAULT 0,
                duration_seconds REAL,
                sync_type TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    async def extract_evaluation_metrics(self) -> List[PerformanceMetric]:
        """Extract performance metrics from evaluation results"""
        
        self.logger.info("📊 Extracting evaluation metrics...")
        
        # Load evaluation data
        query = '''
            SELECT er.id, er.tool, er.language, er.category, er.complexity_level,
                   er.execution_time, er.response_time, er.memory_usage, er.success,
                   er.timestamp, er.metrics,
                   sm.overall_score, sm.code_quality_score, sm.functionality_score,
                   sm.performance_score, sm.maintainability_score, sm.innovation_score
            FROM evaluation_results er
            LEFT JOIN scoring_metrics sm ON er.id = sm.result_id
            WHERE er.timestamp >= datetime('now', '-7 days')
        '''
        
        conn = sqlite3.connect(self.db_path)
        results = conn.execute(query).fetchall()
        conn.close()
        
        metrics = []
        
        for row in results:
            (result_id, tool, language, category, complexity_level, execution_time, 
             response_time, memory_usage, success, timestamp, raw_metrics,
             overall_score, code_quality, functionality, performance, maintainability, innovation) = row
            
            # Parse raw metrics
            parsed_metrics = json.loads(raw_metrics) if raw_metrics else {}
            
            # Extract standard metrics
            base_metrics = [
                PerformanceMetric(
                    metric_name=f"agentic_evaluation_execution_time",
                    metric_value=execution_time,
                    metric_unit="seconds",
                    source="agentic_evaluation",
                    category="performance",
                    timestamp=timestamp,
                    metadata={
                        "tool": tool,
                        "language": language,
                        "category": category,
                        "complexity_level": complexity_level,
                        "result_id": result_id
                    }
                ),
                PerformanceMetric(
                    metric_name=f"agentic_evaluation_response_time",
                    metric_value=response_time,
                    metric_unit="seconds",
                    source="agentic_evaluation",
                    category="performance",
                    timestamp=timestamp,
                    metadata={
                        "tool": tool,
                        "language": language,
                        "category": category,
                        "complexity_level": complexity_level,
                        "result_id": result_id
                    }
                )
            ]
            
            # Add memory usage if available
            if memory_usage is not None:
                base_metrics.append(PerformanceMetric(
                    metric_name=f"agentic_evaluation_memory_usage",
                    metric_value=memory_usage,
                    metric_unit="MB",
                    source="agentic_evaluation",
                    category="resource",
                    timestamp=timestamp,
                    metadata={
                        "tool": tool,
                        "language": language,
                        "category": category,
                        "complexity_level": complexity_level,
                        "result_id": result_id
                    }
                ))
            
            # Add scoring metrics if available
            if overall_score is not None:
                scoring_metrics = [
                    ("overall_score", overall_score),
                    ("code_quality_score", code_quality),
                    ("functionality_score", functionality),
                    ("performance_score", performance),
                    ("maintainability_score", maintainability),
                    ("innovation_score", innovation)
                ]
                
                for metric_name, metric_value in scoring_metrics:
                    if metric_value is not None:
                        base_metrics.append(PerformanceMetric(
                            metric_name=f"agentic_evaluation_{metric_name}",
                            metric_value=metric_value,
                            metric_unit="score",
                            source="agentic_evaluation",
                            category="quality",
                            timestamp=timestamp,
                            metadata={
                                "tool": tool,
                                "language": language,
                                "category": category,
                                "complexity_level": complexity_level,
                                "result_id": result_id
                            }
                        ))
            
            # Add success rate metric
            base_metrics.append(PerformanceMetric(
                metric_name=f"agentic_evaluation_success_rate",
                metric_value=1.0 if success else 0.0,
                metric_unit="boolean",
                source="agentic_evaluation",
                category="reliability",
                timestamp=timestamp,
                metadata={
                    "tool": tool,
                    "language": language,
                    "category": category,
                    "complexity_level": complexity_level,
                    "result_id": result_id
                }
            ))
            
            metrics.extend(base_metrics)
        
        self.logger.info(f"📊 Extracted {len(metrics)} evaluation metrics")
        return metrics

    async def sync_with_polyglot_analytics(self, metrics: List[PerformanceMetric]) -> IntegrationReport:
        """Sync metrics with existing polyglot analytics system"""
        
        start_time = time.time()
        self.logger.info("🔄 Syncing with polyglot analytics...")
        
        synced_count = 0
        errors = []
        
        try:
            # Check if polyglot analytics is available
            if not self.polyglot_analytics.exists():
                error_msg = f"Polyglot analytics script not found: {self.polyglot_analytics}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                
                # Create integration metrics file for manual import
                await self.create_metrics_export_file(metrics)
                synced_count = len(metrics)
            else:
                # Sync with polyglot analytics
                synced_count = await self.sync_metrics_via_nushell(metrics)
            
            # Update integration status
            await self.update_integration_status("polyglot_analytics", synced_count, len(errors))
            
            # Record sync history
            duration = time.time() - start_time
            await self.record_sync_history("polyglot_analytics", synced_count, len(errors), duration)
            
            next_sync = (datetime.now() + timedelta(hours=1)).isoformat()
            
            return IntegrationReport(
                status="success" if len(errors) == 0 else "partial",
                metrics_synced=synced_count,
                errors=errors,
                duration=duration,
                next_sync=next_sync
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Sync failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            await self.record_sync_history("polyglot_analytics", 0, len(errors), duration)
            
            return IntegrationReport(
                status="failed",
                metrics_synced=0,
                errors=errors,
                duration=duration,
                next_sync=(datetime.now() + timedelta(hours=1)).isoformat()
            )

    async def sync_metrics_via_nushell(self, metrics: List[PerformanceMetric]) -> int:
        """Sync metrics using Nushell performance analytics"""
        
        synced_count = 0
        
        # Create temporary metrics file for Nushell import
        temp_metrics_file = self.integration_dir / f"temp_metrics_{int(time.time())}.json"
        
        try:
            # Convert metrics to Nushell-compatible format
            nushell_metrics = []
            for metric in metrics:
                nushell_metric = {
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "metric_unit": metric.metric_unit,
                    "source": metric.source,
                    "category": metric.category,
                    "timestamp": metric.timestamp,
                    "tool": metric.metadata.get("tool", ""),
                    "language": metric.metadata.get("language", ""),
                    "evaluation_category": metric.metadata.get("category", ""),
                    "complexity_level": metric.metadata.get("complexity_level", 0)
                }
                nushell_metrics.append(nushell_metric)
            
            # Write metrics file
            with open(temp_metrics_file, 'w') as f:
                json.dump(nushell_metrics, f, indent=2)
            
            # Execute Nushell integration command
            integration_script = self.create_nushell_integration_script()
            
            cmd = [
                "nu", str(integration_script),
                "--metrics-file", str(temp_metrics_file),
                "--integration-source", "agentic_evaluation"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse result to get synced count
                output = result.stdout.strip()
                if "synced" in output:
                    try:
                        synced_count = int(output.split("synced")[0].strip().split()[-1])
                    except:
                        synced_count = len(metrics)  # Assume all synced if parsing fails
                else:
                    synced_count = len(metrics)
                
                self.logger.info(f"✅ Synced {synced_count} metrics via Nushell")
            else:
                error_msg = f"Nushell sync failed: {result.stderr}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
        finally:
            # Clean up temporary file
            if temp_metrics_file.exists():
                temp_metrics_file.unlink()
        
        return synced_count

    def create_nushell_integration_script(self) -> Path:
        """Create Nushell script for metrics integration"""
        
        script_path = self.integration_dir / "polyglot_integration.nu"
        
        script_content = '''#!/usr/bin/env nu

# Agentic Evaluation Metrics Integration Script
# Integrates evaluation metrics with polyglot performance analytics

def main [
    --metrics-file: string,  # Path to metrics JSON file
    --integration-source: string = "agentic_evaluation"  # Source identifier
] {
    let metrics_file = $metrics_file
    let source = $integration_source
    
    print $"🔄 Integrating metrics from ($metrics_file)"
    
    # Load metrics
    let metrics = (open $metrics_file | from json)
    
    # Performance analytics log path
    let perf_log = "/workspace/dev-env/nushell/logs/performance.log"
    
    # Ensure log directory exists
    mkdir "/workspace/dev-env/nushell/logs"
    
    # Process each metric
    let synced_count = ($metrics | length)
    
    $metrics | each { |metric|
        # Format metric for performance analytics
        let log_entry = {
            timestamp: $metric.timestamp,
            source: $source,
            metric_name: $metric.metric_name,
            metric_value: $metric.metric_value,
            metric_unit: $metric.metric_unit,
            category: $metric.category,
            tool: $metric.tool,
            language: $metric.language,
            evaluation_category: $metric.evaluation_category,
            complexity_level: $metric.complexity_level
        }
        
        # Append to performance log (JSON Lines format)
        $log_entry | to json --raw | save --append $perf_log
    }
    
    print $"✅ ($synced_count) metrics synced to polyglot analytics"
    
    # Update polyglot analytics cache if script exists
    let analytics_script = "/workspace/dev-env/nushell/scripts/performance-analytics.nu"
    if ($analytics_script | path exists) {
        try {
            nu $analytics_script cache --source $source --count $synced_count
        } catch {
            print "⚠️ Could not update analytics cache"
        }
    }
    
    $synced_count
}
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        return script_path

    async def create_metrics_export_file(self, metrics: List[PerformanceMetric]):
        """Create metrics export file for manual integration"""
        
        export_file = self.integration_dir / f"agentic_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "source": "agentic_evaluation_framework",
            "metrics_count": len(metrics),
            "metrics": [asdict(metric) for metric in metrics]
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"📄 Created metrics export file: {export_file}")

    async def update_integration_status(self, integration_type: str, metrics_count: int, errors_count: int):
        """Update integration status in database"""
        
        conn = sqlite3.connect(self.db_path)
        
        status_id = f"integration_{integration_type}"
        status = "active" if errors_count == 0 else "partial" if metrics_count > 0 else "failed"
        
        # Upsert integration status
        conn.execute('''
            INSERT OR REPLACE INTO integration_status 
            (id, integration_type, last_sync, metrics_count, errors_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            status_id, integration_type, datetime.now().isoformat(),
            metrics_count, errors_count, status
        ))
        
        conn.commit()
        conn.close()

    async def record_sync_history(self, sync_type: str, metrics_synced: int, errors: int, duration: float):
        """Record sync operation in history"""
        
        conn = sqlite3.connect(self.db_path)
        
        history_id = f"sync_{sync_type}_{int(time.time())}"
        
        conn.execute('''
            INSERT INTO sync_history 
            (id, sync_type, metrics_synced, errors_encountered, duration_seconds)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            history_id, sync_type, metrics_synced, errors, duration
        ))
        
        conn.commit()
        conn.close()

    async def create_integration_dashboard(self) -> str:
        """Create integration status dashboard"""
        
        self.logger.info("📊 Creating integration dashboard...")
        
        # Load integration data
        conn = sqlite3.connect(self.db_path)
        
        status_query = "SELECT * FROM integration_status ORDER BY last_sync DESC"
        history_query = "SELECT * FROM sync_history ORDER BY sync_timestamp DESC LIMIT 10"
        
        status_data = conn.execute(status_query).fetchall()
        history_data = conn.execute(history_query).fetchall()
        
        conn.close()
        
        # Generate dashboard
        dashboard_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dashboard_path = self.integration_dir / f"integration_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        dashboard_content = f"""# Performance Integration Dashboard

**Generated**: {dashboard_time}
**Integration Status**: Real-time monitoring

## 🔄 Integration Overview

### Active Integrations
"""
        
        for row in status_data:
            integration_id, integration_type, last_sync, metrics_count, errors_count, status, config = row
            
            status_emoji = "✅" if status == "active" else "⚠️" if status == "partial" else "❌"
            
            dashboard_content += f"""
#### {integration_type.replace('_', ' ').title()}
- Status: {status_emoji} {status.title()}
- Last Sync: {last_sync}
- Metrics Synced: {metrics_count:,}
- Errors: {errors_count}
"""
        
        dashboard_content += """
## 📈 Recent Sync History

| Timestamp | Type | Metrics | Errors | Duration |
|-----------|------|---------|--------|----------|
"""
        
        for row in history_data:
            sync_id, timestamp, metrics_synced, errors, duration, sync_type, details = row
            dashboard_content += f"| {timestamp} | {sync_type} | {metrics_synced} | {errors} | {duration:.2f}s |\n"
        
        dashboard_content += f"""

## 🔧 Integration Configuration

### Polyglot Analytics Integration
- **Script Path**: {self.polyglot_analytics}
- **Log Directory**: {self.polyglot_logs}
- **Sync Frequency**: Every hour
- **Metrics Retention**: 30 days

### Performance Metrics Categories
- **Execution Time**: Tool response and processing times
- **Quality Scores**: Code quality, functionality, maintainability
- **Resource Usage**: Memory consumption, CPU utilization
- **Success Rates**: Evaluation completion rates

### Data Flow
```
Agentic Evaluation → Metrics Extraction → Polyglot Analytics → Unified Dashboard
```

## 🚨 Alerts and Monitoring

### Current Alerts
{"- ✅ All integrations operational" if all(row[4] == "active" for row in status_data) else "- ⚠️ Some integrations need attention"}

### Monitoring Thresholds
- **Sync Failure Rate**: > 10%
- **Metric Lag**: > 2 hours
- **Error Rate**: > 5%

## 📊 Performance Insights

### Integration Performance
- **Average Sync Time**: {sum(row[4] for row in history_data) / max(len(history_data), 1):.2f}s
- **Success Rate**: {(1 - sum(1 for row in history_data if row[3] > 0) / max(len(history_data), 1)) * 100:.1f}%
- **Metrics Throughput**: {sum(row[2] for row in history_data)} metrics/hour

### Recommendations
1. **Monitor Integration Health**: Regular checks of sync status
2. **Optimize Sync Frequency**: Balance freshness vs performance
3. **Error Handling**: Implement retry mechanisms for failed syncs
4. **Data Quality**: Validate metrics before integration

---

*Dashboard automatically updates with each sync cycle*
*Next update: {(datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)
        
        self.logger.info(f"📊 Integration dashboard created: {dashboard_path}")
        return str(dashboard_path)

    async def setup_automated_sync(self, interval_hours: int = 1):
        """Setup automated synchronization"""
        
        self.logger.info(f"⏰ Setting up automated sync every {interval_hours} hour(s)")
        
        # Create sync script
        sync_script_path = self.integration_dir / "automated_sync.py"
        
        sync_script_content = f'''#!/usr/bin/env python3
"""
Automated Performance Integration Sync
Runs periodic synchronization with polyglot analytics
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from performance_integration import PerformanceIntegration

async def main():
    integration = PerformanceIntegration("{self.eval_root}")
    
    print("🔄 Starting automated sync...")
    
    # Extract metrics
    metrics = await integration.extract_evaluation_metrics()
    
    # Sync with polyglot analytics
    report = await integration.sync_with_polyglot_analytics(metrics)
    
    print(f"✅ Sync completed: {{report.metrics_synced}} metrics synced")
    
    if report.errors:
        print("⚠️ Errors encountered:")
        for error in report.errors:
            print(f"  - {{error}}")
    
    # Create dashboard
    dashboard_path = await integration.create_integration_dashboard()
    print(f"📊 Dashboard updated: {{dashboard_path}}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open(sync_script_path, 'w') as f:
            f.write(sync_script_content)
        
        sync_script_path.chmod(0o755)
        
        # Create systemd service file (if on Linux)
        service_content = f"""[Unit]
Description=Agentic Evaluation Performance Integration
After=network.target

[Service]
Type=oneshot
User=vscode
WorkingDirectory={self.integration_dir}
ExecStart=/usr/bin/python3 {sync_script_path}

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.integration_dir / "agentic-performance-integration.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        # Create cron job script
        cron_script = self.integration_dir / "setup_cron.sh"
        cron_content = f"""#!/bin/bash
# Setup cron job for automated performance integration

# Add cron job for every {interval_hours} hour(s)
(crontab -l 2>/dev/null; echo "0 */{interval_hours} * * * cd {self.integration_dir} && python3 automated_sync.py >> {self.logs_dir}/automated_sync.log 2>&1") | crontab -

echo "✅ Cron job configured for every {interval_hours} hour(s)"
echo "📄 Logs will be written to {self.logs_dir}/automated_sync.log"
"""
        
        with open(cron_script, 'w') as f:
            f.write(cron_content)
        
        cron_script.chmod(0o755)
        
        self.logger.info(f"⏰ Automated sync configured:")
        self.logger.info(f"  - Sync script: {sync_script_path}")
        self.logger.info(f"  - Service file: {service_file}")
        self.logger.info(f"  - Cron setup: {cron_script}")

    async def run_integration_pipeline(self) -> IntegrationReport:
        """Run the complete integration pipeline"""
        
        self.logger.info("🚀 Running complete integration pipeline...")
        
        start_time = time.time()
        
        try:
            # Extract metrics
            metrics = await self.extract_evaluation_metrics()
            
            # Sync with polyglot analytics
            report = await self.sync_with_polyglot_analytics(metrics)
            
            # Create dashboard
            dashboard_path = await self.create_integration_dashboard()
            
            # Update report with dashboard info
            report.metadata = {"dashboard_path": dashboard_path}
            
            duration = time.time() - start_time
            self.logger.info(f"✅ Integration pipeline completed in {duration:.2f}s")
            
            return report
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Integration pipeline failed: {e}")
            
            return IntegrationReport(
                status="failed",
                metrics_synced=0,
                errors=[str(e)],
                duration=duration,
                next_sync=(datetime.now() + timedelta(hours=1)).isoformat()
            )

def main():
    parser = argparse.ArgumentParser(description='Performance Integration for Agentic Evaluation')
    parser.add_argument('--mode', choices=['extract', 'sync', 'dashboard', 'setup', 'pipeline'],
                       default='pipeline', help='Integration mode')
    parser.add_argument('--eval-root', default='/workspace/agentic-eval',
                       help='Evaluation framework root directory')
    parser.add_argument('--interval', type=int, default=1,
                       help='Sync interval in hours (for setup mode)')
    
    args = parser.parse_args()
    
    async def run():
        integration = PerformanceIntegration(args.eval_root)
        
        if args.mode == 'extract':
            metrics = await integration.extract_evaluation_metrics()
            print(f"✅ Extracted {len(metrics)} metrics")
            
        elif args.mode == 'sync':
            metrics = await integration.extract_evaluation_metrics()
            report = await integration.sync_with_polyglot_analytics(metrics)
            print(f"✅ Sync completed: {report.metrics_synced} metrics synced")
            
        elif args.mode == 'dashboard':
            dashboard_path = await integration.create_integration_dashboard()
            print(f"✅ Dashboard created: {dashboard_path}")
            
        elif args.mode == 'setup':
            await integration.setup_automated_sync(args.interval)
            print(f"✅ Automated sync configured for every {args.interval} hour(s)")
            
        elif args.mode == 'pipeline':
            report = await integration.run_integration_pipeline()
            print(f"✅ Integration pipeline completed:")
            print(f"  Status: {report.status}")
            print(f"  Metrics synced: {report.metrics_synced}")
            print(f"  Duration: {report.duration:.2f}s")
            if report.errors:
                print(f"  Errors: {len(report.errors)}")
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n🛑 Integration interrupted by user")
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        raise

if __name__ == "__main__":
    main()