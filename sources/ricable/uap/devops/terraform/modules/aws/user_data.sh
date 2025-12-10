#!/bin/bash
# UAP Application Server User Data Script
# Configures EC2 instances for UAP deployment

set -e

# Variables from Terraform
ENVIRONMENT=${environment}
PROJECT=${project}

# Log file for debugging
LOG_FILE="/var/log/uap-setup.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=== UAP Server Setup Started: $(date) ==="
echo "Environment: $ENVIRONMENT"
echo "Project: $PROJECT"

# Update system packages
echo "Updating system packages..."
yum update -y

# Install required packages
echo "Installing required packages..."
yum install -y \
    git \
    curl \
    wget \
    unzip \
    htop \
    jq \
    docker \
    amazon-cloudwatch-agent \
    awscli

# Start and enable Docker
echo "Starting Docker service..."
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
echo "Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Node.js 20 (for frontend)
echo "Installing Node.js 20..."
curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
yum install -y nodejs

# Install Python 3.11 (for backend)
echo "Installing Python 3.11..."
yum install -y python3.11 python3.11-pip python3.11-venv

# Create application directory
echo "Creating application directory..."
mkdir -p /opt/uap
chown ec2-user:ec2-user /opt/uap

# Create log directory
echo "Creating log directory..."
mkdir -p /var/log/uap
chown ec2-user:ec2-user /var/log/uap

# Download and setup UAP application (placeholder)
echo "Setting up UAP application..."
cd /opt/uap

# Clone repository (in production, this would be from a specific release)
# git clone https://github.com/your-org/uap.git .

# For now, create a basic health check endpoint
cat > /opt/uap/health_check.py << 'EOF'
#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import datetime

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            response = {
                'status': 'healthy',
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'environment': '${environment}',
                'project': '${project}',
                'version': '1.0.0'
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8000), HealthCheckHandler)
    print('Health check server starting on port 8000...')
    server.serve_forever()
EOF

chmod +x /opt/uap/health_check.py

# Create systemd service for UAP application
echo "Creating systemd service..."
cat > /etc/systemd/system/uap-health.service << EOF
[Unit]
Description=UAP Health Check Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/uap
ExecStart=/usr/bin/python3 /opt/uap/health_check.py
Restart=always
RestartSec=10
Environment=ENVIRONMENT=$ENVIRONMENT
Environment=PROJECT=$PROJECT

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl daemon-reload
systemctl enable uap-health
systemctl start uap-health

# Configure CloudWatch Agent
echo "Configuring CloudWatch Agent..."
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/uap/*.log",
                        "log_group_name": "/aws/ec2/$PROJECT-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/uap-application"
                    },
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/ec2/$PROJECT-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/system-messages"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "UAP/Application",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch Agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Configure automatic security updates
echo "Configuring automatic security updates..."
yum install -y yum-cron
sed -i 's/update_cmd = default/update_cmd = security/' /etc/yum/yum-cron.conf
sed -i 's/apply_updates = no/apply_updates = yes/' /etc/yum/yum-cron.conf
systemctl enable yum-cron
systemctl start yum-cron

# Set up log rotation for application logs
echo "Configuring log rotation..."
cat > /etc/logrotate.d/uap << EOF
/var/log/uap/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su ec2-user ec2-user
}
EOF

# Install and configure fail2ban for security
echo "Installing and configuring fail2ban..."
yum install -y epel-release
yum install -y fail2ban

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/secure
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure firewall (basic rules)
echo "Configuring firewall..."
systemctl enable firewalld
systemctl start firewalld

# Allow SSH and application ports
firewall-cmd --permanent --add-service=ssh
firewall-cmd --permanent --add-port=8000/tcp
firewall-cmd --reload

# Create monitoring script
echo "Creating monitoring script..."
cat > /opt/uap/monitor.sh << 'EOF'
#!/bin/bash
# UAP Application Monitoring Script

LOG_FILE="/var/log/uap/monitor.log"
HEALTH_URL="http://localhost:8000/health"

# Check if application is responding
if curl -f -s $HEALTH_URL > /dev/null; then
    echo "$(date): Health check passed" >> $LOG_FILE
else
    echo "$(date): Health check failed - restarting service" >> $LOG_FILE
    systemctl restart uap-health
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    echo "$(date): High disk usage: ${DISK_USAGE}%" >> $LOG_FILE
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
if (( $(echo "$MEM_USAGE > 85" | bc -l) )); then
    echo "$(date): High memory usage: ${MEM_USAGE}%" >> $LOG_FILE
fi
EOF

chmod +x /opt/uap/monitor.sh

# Add monitoring to crontab
echo "Adding monitoring to crontab..."
(crontab -u ec2-user -l 2>/dev/null; echo "*/5 * * * * /opt/uap/monitor.sh") | crontab -u ec2-user -

# Create startup script for UAP application
echo "Creating startup script..."
cat > /opt/uap/start.sh << 'EOF'
#!/bin/bash
# UAP Application Startup Script

cd /opt/uap

# Source environment variables
export ENVIRONMENT=${environment}
export PROJECT=${project}

# Start the application
echo "Starting UAP application..."
python3 health_check.py
EOF

chmod +x /opt/uap/start.sh

# Final system configuration
echo "Final system configuration..."

# Set timezone
timedatectl set-timezone UTC

# Configure SSH hardening
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Set up swap file if not exists
if [ ! -f /swapfile ]; then
    echo "Creating swap file..."
    dd if=/dev/zero of=/swapfile bs=1024 count=2097152  # 2GB
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
fi

# Signal CloudFormation that setup is complete
echo "Signaling completion..."
/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource AutoScalingGroup --region ${AWS::Region} || true

echo "=== UAP Server Setup Completed: $(date) ==="
echo "Application health check: http://localhost:8000/health"