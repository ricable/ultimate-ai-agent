// Advanced system configuration management interface
import React, { useState, useEffect } from 'react';
import { Settings, Database, Shield, Zap, Globe, AlertTriangle, CheckCircle, Save, RotateCcw, Download, Upload, Eye, EyeOff } from 'lucide-react';

interface ConfigSection {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  settings: ConfigSetting[];
}

interface ConfigSetting {
  id: string;
  name: string;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'textarea' | 'password';
  value: any;
  default_value: any;
  options?: string[];
  validation?: {
    required?: boolean;
    min?: number;
    max?: number;
    pattern?: string;
  };
  sensitive?: boolean;
  restart_required?: boolean;
}

interface ConfigBackup {
  id: string;
  name: string;
  timestamp: string;
  size: number;
  created_by: string;
  auto_generated: boolean;
}

export const SystemConfiguration: React.FC = () => {
  const [configSections, setConfigSections] = useState<ConfigSection[]>([]);
  const [activeSection, setActiveSection] = useState<string>('performance');
  const [hasChanges, setHasChanges] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [showSensitive, setShowSensitive] = useState<Record<string, boolean>>({});
  const [backups, setBackups] = useState<ConfigBackup[]>([]);
  const [showBackups, setShowBackups] = useState(false);

  // Mock configuration data
  const mockConfigSections: ConfigSection[] = [
    {
      id: 'performance',
      name: 'Performance & Scaling',
      description: 'System performance and auto-scaling settings',
      icon: Zap,
      settings: [
        {
          id: 'max_concurrent_connections',
          name: 'Max Concurrent Connections',
          description: 'Maximum number of simultaneous WebSocket connections',
          type: 'number',
          value: 1000,
          default_value: 1000,
          validation: { required: true, min: 1, max: 10000 },
          restart_required: true
        },
        {
          id: 'response_timeout',
          name: 'Response Timeout (ms)',
          description: 'Maximum time to wait for agent responses',
          type: 'number',
          value: 30000,
          default_value: 30000,
          validation: { required: true, min: 1000, max: 300000 }
        },
        {
          id: 'auto_scaling_enabled',
          name: 'Auto Scaling',
          description: 'Enable automatic resource scaling based on demand',
          type: 'boolean',
          value: true,
          default_value: true
        },
        {
          id: 'scaling_threshold',
          name: 'Scaling Threshold (%)',
          description: 'CPU usage threshold to trigger scaling',
          type: 'number',
          value: 80,
          default_value: 80,
          validation: { required: true, min: 10, max: 95 }
        }
      ]
    },
    {
      id: 'database',
      name: 'Database Configuration',
      description: 'Database connection and performance settings',
      icon: Database,
      settings: [
        {
          id: 'database_url',
          name: 'Database URL',
          description: 'PostgreSQL connection string',
          type: 'password',
          value: 'postgresql://user:****@localhost:5432/uap',
          default_value: 'postgresql://localhost:5432/uap',
          sensitive: true,
          restart_required: true
        },
        {
          id: 'connection_pool_size',
          name: 'Connection Pool Size',
          description: 'Maximum number of database connections',
          type: 'number',
          value: 20,
          default_value: 20,
          validation: { required: true, min: 1, max: 100 }
        },
        {
          id: 'query_timeout',
          name: 'Query Timeout (ms)',
          description: 'Maximum time for database queries',
          type: 'number',
          value: 10000,
          default_value: 10000,
          validation: { required: true, min: 1000, max: 60000 }
        },
        {
          id: 'enable_query_logging',
          name: 'Enable Query Logging',
          description: 'Log all database queries for debugging',
          type: 'boolean',
          value: false,
          default_value: false
        }
      ]
    },
    {
      id: 'security',
      name: 'Security & Authentication',
      description: 'Security policies and authentication settings',
      icon: Shield,
      settings: [
        {
          id: 'jwt_secret',
          name: 'JWT Secret Key',
          description: 'Secret key for JWT token signing',
          type: 'password',
          value: '********************************',
          default_value: '',
          sensitive: true,
          restart_required: true,
          validation: { required: true }
        },
        {
          id: 'jwt_expiry',
          name: 'JWT Token Expiry (minutes)',
          description: 'JWT token expiration time',
          type: 'number',
          value: 30,
          default_value: 30,
          validation: { required: true, min: 5, max: 1440 }
        },
        {
          id: 'password_policy',
          name: 'Password Policy',
          description: 'Password complexity requirements',
          type: 'select',
          value: 'strong',
          default_value: 'medium',
          options: ['weak', 'medium', 'strong', 'custom']
        },
        {
          id: 'require_2fa',
          name: 'Require 2FA',
          description: 'Require two-factor authentication for all users',
          type: 'boolean',
          value: false,
          default_value: false
        },
        {
          id: 'session_timeout',
          name: 'Session Timeout (minutes)',
          description: 'Automatic logout after inactivity',
          type: 'number',
          value: 60,
          default_value: 60,
          validation: { required: true, min: 5, max: 480 }
        }
      ]
    },
    {
      id: 'api',
      name: 'API Configuration',
      description: 'External API integrations and settings',
      icon: Globe,
      settings: [
        {
          id: 'openai_api_key',
          name: 'OpenAI API Key',
          description: 'API key for OpenAI services',
          type: 'password',
          value: 'sk-proj-********************************',
          default_value: '',
          sensitive: true,
          validation: { required: true }
        },
        {
          id: 'anthropic_api_key',
          name: 'Anthropic API Key',
          description: 'API key for Claude services',
          type: 'password',
          value: 'sk-ant-********************************',
          default_value: '',
          sensitive: true,
          validation: { required: true }
        },
        {
          id: 'api_rate_limit',
          name: 'API Rate Limit (requests/min)',
          description: 'Rate limit for external API calls',
          type: 'number',
          value: 100,
          default_value: 100,
          validation: { required: true, min: 1, max: 1000 }
        },
        {
          id: 'api_timeout',
          name: 'API Timeout (ms)',
          description: 'Timeout for external API requests',
          type: 'number',
          value: 30000,
          default_value: 30000,
          validation: { required: true, min: 1000, max: 120000 }
        }
      ]
    },
    {
      id: 'monitoring',
      name: 'Monitoring & Logging',
      description: 'System monitoring and logging configuration',
      icon: Settings,
      settings: [
        {
          id: 'log_level',
          name: 'Log Level',
          description: 'Minimum log level to record',
          type: 'select',
          value: 'INFO',
          default_value: 'INFO',
          options: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        },
        {
          id: 'enable_metrics',
          name: 'Enable Metrics Collection',
          description: 'Collect Prometheus metrics',
          type: 'boolean',
          value: true,
          default_value: true
        },
        {
          id: 'metrics_retention_days',
          name: 'Metrics Retention (days)',
          description: 'How long to keep metrics data',
          type: 'number',
          value: 30,
          default_value: 30,
          validation: { required: true, min: 1, max: 365 }
        },
        {
          id: 'enable_audit_log',
          name: 'Enable Audit Logging',
          description: 'Log all user actions for compliance',
          type: 'boolean',
          value: true,
          default_value: true
        }
      ]
    }
  ];

  const mockBackups: ConfigBackup[] = [
    {
      id: '1',
      name: 'Auto Backup - 2024-12-28',
      timestamp: '2024-12-28T10:00:00Z',
      size: 24576,
      created_by: 'system',
      auto_generated: true
    },
    {
      id: '2',
      name: 'Pre-deployment Backup',
      timestamp: '2024-12-27T15:30:00Z',
      size: 23890,
      created_by: 'admin',
      auto_generated: false
    },
    {
      id: '3',
      name: 'Auto Backup - 2024-12-27',
      timestamp: '2024-12-27T10:00:00Z',
      size: 23654,
      created_by: 'system',
      auto_generated: true
    }
  ];

  useEffect(() => {
    setTimeout(() => {
      setConfigSections(mockConfigSections);
      setBackups(mockBackups);
      setIsLoading(false);
    }, 1000);
  }, []);

  const handleSettingChange = (sectionId: string, settingId: string, newValue: any) => {
    setConfigSections(prev => prev.map(section => {
      if (section.id === sectionId) {
        return {
          ...section,
          settings: section.settings.map(setting => 
            setting.id === settingId ? { ...setting, value: newValue } : setting
          )
        };
      }
      return section;
    }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save configuration:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    if (confirm('Are you sure you want to reset all settings to their default values?')) {
      setConfigSections(prev => prev.map(section => ({
        ...section,
        settings: section.settings.map(setting => ({
          ...setting,
          value: setting.default_value
        }))
      })));
      setHasChanges(true);
    }
  };

  const toggleSensitiveVisibility = (settingId: string) => {
    setShowSensitive(prev => ({
      ...prev,
      [settingId]: !prev[settingId]
    }));
  };

  const exportConfig = () => {
    const configData = {
      exported_at: new Date().toISOString(),
      sections: configSections.map(section => ({
        id: section.id,
        name: section.name,
        settings: section.settings.map(setting => ({
          id: setting.id,
          name: setting.name,
          value: setting.sensitive ? '[SENSITIVE]' : setting.value,
          type: setting.type
        }))
      }))
    };
    
    const blob = new Blob([JSON.stringify(configData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `uap-config-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const activeConfigSection = configSections.find(section => section.id === activeSection);

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="flex space-x-4">
            <div className="w-64 bg-gray-200 rounded-lg h-96"></div>
            <div className="flex-1 bg-gray-200 rounded-lg h-96"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Configuration</h1>
          <p className="text-gray-500 mt-1">Manage system settings and preferences</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowBackups(!showBackups)}
            className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            <Database className="h-4 w-4 mr-2" />
            Backups
          </button>
          <button
            onClick={exportConfig}
            className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </button>
          <button
            onClick={handleReset}
            className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || isSaving}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <Save className="h-4 w-4 mr-2" />
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Changes Indicator */}
      {hasChanges && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2" />
            <div>
              <p className="text-sm font-medium text-yellow-800">
                You have unsaved changes
              </p>
              <p className="text-xs text-yellow-600">
                Don't forget to save your configuration changes before leaving this page.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Configuration Interface */}
      <div className="flex space-x-6">
        {/* Sidebar */}
        <div className="w-64 bg-white rounded-lg border border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">Configuration Sections</h3>
          </div>
          <nav className="p-2">
            {configSections.map((section) => {
              const Icon = section.icon;
              const hasChangesInSection = section.settings.some(setting => 
                setting.value !== setting.default_value
              );
              
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center px-3 py-2 text-left rounded-lg transition-colors mb-1 ${
                    activeSection === section.id
                      ? 'bg-blue-50 text-blue-700 border-blue-200'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Icon className={`h-4 w-4 mr-3 ${
                    activeSection === section.id ? 'text-blue-500' : 'text-gray-400'
                  }`} />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{section.name}</span>
                      {hasChangesInSection && (
                        <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{section.description}</p>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1">
          {activeConfigSection && (
            <div className="bg-white rounded-lg border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex items-center">
                  <activeConfigSection.icon className="h-6 w-6 text-blue-500 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold">{activeConfigSection.name}</h3>
                    <p className="text-sm text-gray-500">{activeConfigSection.description}</p>
                  </div>
                </div>
              </div>
              
              <div className="p-6 space-y-6">
                {activeConfigSection.settings.map((setting) => {
                  const hasChanged = setting.value !== setting.default_value;
                  const isVisible = showSensitive[setting.id] || !setting.sensitive;
                  
                  return (
                    <div key={setting.id} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
                      <div>
                        <label className="text-sm font-medium text-gray-900 flex items-center">
                          {setting.name}
                          {hasChanged && (
                            <div className="w-2 h-2 bg-yellow-500 rounded-full ml-2"></div>
                          )}
                          {setting.restart_required && (
                            <AlertTriangle className="h-4 w-4 text-orange-500 ml-2" title="Restart required" />
                          )}
                        </label>
                        <p className="text-xs text-gray-500 mt-1">{setting.description}</p>
                        {setting.validation?.required && (
                          <p className="text-xs text-red-500 mt-1">* Required</p>
                        )}
                      </div>
                      
                      <div className="md:col-span-1">
                        {setting.type === 'boolean' ? (
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              checked={setting.value as boolean}
                              onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, e.target.checked)}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="ml-2 text-sm text-gray-600">
                              {setting.value ? 'Enabled' : 'Disabled'}
                            </span>
                          </label>
                        ) : setting.type === 'select' ? (
                          <select
                            value={setting.value as string}
                            onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          >
                            {setting.options?.map(option => (
                              <option key={option} value={option}>{option}</option>
                            ))}
                          </select>
                        ) : setting.type === 'textarea' ? (
                          <textarea
                            value={setting.value as string}
                            onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                            rows={3}
                          />
                        ) : setting.type === 'password' ? (
                          <div className="relative">
                            <input
                              type={isVisible ? 'text' : 'password'}
                              value={setting.value as string}
                              onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, e.target.value)}
                              className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                            />
                            <button
                              type="button"
                              onClick={() => toggleSensitiveVisibility(setting.id)}
                              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                            >
                              {isVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                            </button>
                          </div>
                        ) : setting.type === 'number' ? (
                          <input
                            type="number"
                            value={setting.value as number}
                            onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, Number(e.target.value))}
                            min={setting.validation?.min}
                            max={setting.validation?.max}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                        ) : (
                          <input
                            type="text"
                            value={setting.value as string}
                            onChange={(e) => handleSettingChange(activeConfigSection.id, setting.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                          />
                        )}
                      </div>
                      
                      <div className="text-xs text-gray-500">
                        {setting.sensitive && '🔒 Sensitive'}
                        {setting.restart_required && (
                          <div className="mt-1 text-orange-600">🔄 Restart Required</div>
                        )}
                        {setting.validation && (
                          <div className="mt-1">
                            {setting.validation.min !== undefined && setting.validation.max !== undefined && (
                              <div>Range: {setting.validation.min} - {setting.validation.max}</div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Backups Modal */}
      {showBackups && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h3 className="text-lg font-semibold">Configuration Backups</h3>
              <button
                onClick={() => setShowBackups(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
            
            <div className="p-6">
              <div className="space-y-4">
                {backups.map((backup) => (
                  <div key={backup.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                    <div>
                      <div className="font-medium">{backup.name}</div>
                      <div className="text-sm text-gray-500">
                        {new Date(backup.timestamp).toLocaleString()} • {formatBytes(backup.size)} • {backup.created_by}
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {backup.auto_generated && (
                        <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">Auto</span>
                      )}
                      <button className="text-blue-600 hover:text-blue-800 text-sm">Restore</button>
                      <button className="text-gray-600 hover:text-gray-800 text-sm">Download</button>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 pt-4 border-t border-gray-200">
                <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                  <Save className="h-4 w-4 mr-2" />
                  Create Backup
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};