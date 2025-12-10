#!/bin/bash

# Host Shell Integration for Polyglot Development Environment
# Source this file in your shell configuration (.bashrc, .zshrc, etc.)
# These aliases run on the HOST machine, not inside containers

# Host environment navigation
alias polyglot-root="cd /Users/cedric/dev/github.com/polyglot-devenv"
alias host-tools="cd /Users/cedric/dev/github.com/polyglot-devenv/host-tooling"

# DevPod management (HOST-side container management)
alias devpod-status="devpod list"
alias devpod-cleanup="nu host-tooling/devpod-management/devpod-manage.nu cleanup --all"
alias devpod-provision-python="nu host-tooling/devpod-management/devpod-provision.nu python"
alias devpod-provision-typescript="nu host-tooling/devpod-management/devpod-provision.nu typescript"
alias devpod-provision-rust="nu host-tooling/devpod-management/devpod-provision.nu rust"
alias devpod-provision-go="nu host-tooling/devpod-management/devpod-provision.nu go"

# Infrastructure management (requires HOST credentials)
alias k8s-status="nu host-tooling/monitoring/kubernetes.nu status"
alias k8s-deploy="nu host-tooling/monitoring/kubernetes.nu deploy"
alias github-status="nu host-tooling/monitoring/github.nu status"

# Docker management (HOST Docker daemon)
alias docker-setup="nu host-tooling/installation/docker-setup.nu --install --configure --optimize"
alias docker-status="nu host-tooling/installation/docker-setup.nu --status"
alias docker-reset="nu host-tooling/installation/docker-setup.nu --reset"

# Host system monitoring
alias host-resources="docker stats"
alias host-containers="docker ps -a"
alias host-volumes="docker volume ls"

# Enter container environments (HOST command to enter containers)
alias enter-python="devpod ssh polyglot-python-devpod-\$(date +%Y%m%d)-1 || echo 'No Python container found. Run: devpod-provision-python'"
alias enter-typescript="devpod ssh polyglot-typescript-devpod-\$(date +%Y%m%d)-1 || echo 'No TypeScript container found. Run: devpod-provision-typescript'"
alias enter-rust="devpod ssh polyglot-rust-devpod-\$(date +%Y%m%d)-1 || echo 'No Rust container found. Run: devpod-provision-rust'"
alias enter-go="devpod ssh polyglot-go-devpod-\$(date +%Y%m%d)-1 || echo 'No Go container found. Run: devpod-provision-go'"

# Host backup and recovery
alias backup-configs="tar -czf ~/polyglot-configs-\$(date +%Y%m%d).tar.gz host-tooling/ .claude/ devpod-automation/config/"
alias list-backups="ls -la ~/polyglot-*.tar.gz"

# Host development workflow
alias polyglot-init="nu host-tooling/installation/docker-setup.nu --install --configure && echo 'Ready to provision containers'"
alias polyglot-health="nu host-tooling/monitoring/kubernetes.nu status && docker system df"

echo "✅ Polyglot host tooling aliases loaded"
echo "🏠 Host commands: polyglot-root, host-tools, docker-setup, devpod-status"
echo "🐳 DevPod management: devpod-provision-*, enter-*, devpod-cleanup"
echo "☁️  Infrastructure: k8s-*, github-status (requires host credentials)"
echo "💾 Backup: backup-configs, list-backups"