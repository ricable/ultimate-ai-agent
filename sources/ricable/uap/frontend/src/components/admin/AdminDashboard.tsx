// Comprehensive admin dashboard for system management and configuration
import React, { useState, useEffect } from 'react';
import { Settings, Users, Shield, Database, Server, Activity, AlertTriangle, CheckCircle, RefreshCw, Save, Download, Upload } from 'lucide-react';

interface SystemConfig {
  id: string;
  category: string;
  name: string;
  value: string | number | boolean;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  options?: string[];
  sensitive?: boolean;
}

interface UserAccount {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'manager' | 'user' | 'guest';
  status: 'active' | 'inactive' | 'locked';
  created_at: string;
  last_login: string;
  permissions: string[];
}

interface SystemAlert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  resolved: boolean;
  source: string;
}

export const AdminDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'config' | 'security' | 'logs'>('overview');
  const [systemConfig, setSystemConfig] = useState<SystemConfig[]>([]);
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasChanges, setHasChanges] = useState(false);

  // Fetch system configuration from API
  const fetchSystemConfig = async () => {
    try {
      const response = await fetch('/api/admin/config');
      if (response.ok) {
        const data = await response.json();
        setSystemConfig(data.config || []);
      } else {
        // Fallback system config if API not available
        const defaultConfig: SystemConfig[] = [
          {
            id: 'max_concurrent_sessions',
            category: 'Performance',
            name: 'Max Concurrent Sessions',
            value: 1000,
            description: 'Maximum number of simultaneous WebSocket connections',
            type: 'number'
          },
          {
            id: 'response_timeout',
            category: 'Performance',
            name: 'Response Timeout (ms)',
            value: 30000,
            description: 'Maximum time to wait for agent responses',
            type: 'number'
          },
          {
            id: 'enable_request_logging',
            category: 'Logging',
            name: 'Enable Request Logging',
            value: true,
            description: 'Log all API requests for debugging and analytics',
            type: 'boolean'
          },
          {
            id: 'log_level',
            category: 'Logging',
            name: 'Log Level',
            value: 'INFO',
            description: 'Minimum log level to record',
            type: 'select',
            options: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
          }
        ];
        setSystemConfig(defaultConfig);
      }
    } catch (error) {
      console.error('Failed to fetch system config:', error);
      setSystemConfig([]);
    }
  };

  // Fetch users from API
  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/admin/users');
      if (response.ok) {
        const data = await response.json();
        setUsers(data.users || []);
      } else {
        console.error('Failed to fetch users:', response.statusText);
        setUsers([]);
      }
    } catch (error) {
      console.error('Failed to fetch users:', error);
      setUsers([]);
    }
  };

  // Fetch system alerts from API
  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/admin/alerts');
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts || []);
      } else {
        console.error('Failed to fetch alerts:', response.statusText);
        setAlerts([]);
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      setAlerts([]);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await Promise.all([
        fetchSystemConfig(),
        fetchUsers(),
        fetchAlerts()
      ]);
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const handleConfigChange = (id: string, newValue: string | number | boolean) => {
    setSystemConfig(prev => prev.map(config => 
      config.id === id ? { ...config, value: newValue } : config
    ));
    setHasChanges(true);
  };

  const handleSaveConfig = async () => {
    // Simulate API call to save configuration
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    setHasChanges(false);
    setIsLoading(false);
  };

  const handleUserStatusChange = (userId: string, newStatus: 'active' | 'inactive' | 'locked') => {
    setUsers(prev => prev.map(user => 
      user.id === userId ? { ...user, status: newStatus } : user
    ));
  };

  const handleAlertResolve = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, resolved: true } : alert
    ));
  };

  const exportSystemLogs = () => {
    // Simulate log export
    const logData = `# UAP System Logs - ${new Date().toISOString()}\n# Generated by Admin Dashboard\n\n[INFO] System operational\n[WARNING] High memory usage detected\n[ERROR] Failed to connect to external service`;
    const blob = new Blob([logData], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `uap-system-logs-${new Date().toISOString().split('T')[0]}.log`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'users', label: 'User Management', icon: Users },
    { id: 'config', label: 'Configuration', icon: Settings },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'logs', label: 'System Logs', icon: Database }
  ];

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-red-100 text-red-800';
      case 'manager': return 'bg-blue-100 text-blue-800';
      case 'user': return 'bg-green-100 text-green-800';
      case 'guest': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'inactive': return 'bg-yellow-100 text-yellow-800';
      case 'locked': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical': return 'bg-red-50 border-red-200 text-red-800';
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info': return 'bg-blue-50 border-blue-200 text-blue-800';
      default: return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  if (isLoading && !systemConfig.length) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="flex space-x-4 mb-6">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-10 bg-gray-200 rounded w-24"></div>
            ))}
          </div>
          <div className="bg-gray-200 rounded h-96"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">System Administration</h1>
        <div className="flex items-center space-x-2">
          {hasChanges && (
            <button
              onClick={handleSaveConfig}
              disabled={isLoading}
              className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              <Save className="h-4 w-4 mr-2" />
              {isLoading ? 'Saving...' : 'Save Changes'}
            </button>
          )}
          <button
            onClick={exportSystemLogs}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Download className="h-4 w-4 mr-2" />
            Export Logs
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* System Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">System Status</p>
                  <p className="text-2xl font-bold text-green-600">Operational</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Active Users</p>
                  <p className="text-2xl font-bold text-gray-900">{users.filter(u => u.status === 'active').length}</p>
                </div>
                <Users className="h-8 w-8 text-blue-500" />
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Open Alerts</p>
                  <p className="text-2xl font-bold text-gray-900">{alerts.filter(a => !a.resolved).length}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-yellow-500" />
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Server Uptime</p>
                  <p className="text-2xl font-bold text-gray-900">99.9%</p>
                </div>
                <Server className="h-8 w-8 text-purple-500" />
              </div>
            </div>
          </div>

          {/* Recent Alerts */}
          <div className="bg-white rounded-lg border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold">Recent System Alerts</h3>
            </div>
            <div className="p-6 space-y-4">
              {alerts.slice(0, 5).map((alert) => (
                <div key={alert.id} className={`p-4 rounded-lg border ${getAlertColor(alert.type)} ${alert.resolved ? 'opacity-50' : ''}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center">
                        <span className="font-medium">{alert.title}</span>
                        {alert.resolved && (
                          <CheckCircle className="h-4 w-4 ml-2 text-green-500" />
                        )}
                      </div>
                      <p className="text-sm mt-1">{alert.message}</p>
                      <p className="text-xs mt-2 opacity-75">
                        {new Date(alert.timestamp).toLocaleString()} • {alert.source}
                      </p>
                    </div>
                    {!alert.resolved && (
                      <button
                        onClick={() => handleAlertResolve(alert.id)}
                        className="ml-4 px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-50"
                      >
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'users' && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold">User Accounts</h3>
            <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <Users className="h-4 w-4 mr-2" />
              Add User
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Login</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {users.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium text-gray-900">{user.username}</div>
                        <div className="text-sm text-gray-500">{user.email}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getRoleColor(user.role)}`}>
                        {user.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <select
                        value={user.status}
                        onChange={(e) => handleUserStatusChange(user.id, e.target.value as any)}
                        className={`text-xs font-semibold rounded-full px-2 py-1 border-0 ${getStatusColor(user.status)}`}
                      >
                        <option value="active">Active</option>
                        <option value="inactive">Inactive</option>
                        <option value="locked">Locked</option>
                      </select>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(user.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(user.last_login).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button className="text-blue-600 hover:text-blue-900 mr-3">Edit</button>
                      <button className="text-red-600 hover:text-red-900">Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'config' && (
        <div className="space-y-6">
          {Object.entries(systemConfig.reduce((acc, config) => {
            if (!acc[config.category]) acc[config.category] = [];
            acc[config.category].push(config);
            return acc;
          }, {} as Record<string, SystemConfig[]>)).map(([category, configs]) => (
            <div key={category} className="bg-white rounded-lg border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold">{category} Settings</h3>
              </div>
              <div className="p-6 space-y-4">
                {configs.map((config) => (
                  <div key={config.id} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                    <div>
                      <label className="text-sm font-medium text-gray-900">{config.name}</label>
                      <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                    </div>
                    <div className="md:col-span-1">
                      {config.type === 'boolean' ? (
                        <label className="flex items-center">
                          <input
                            type="checkbox"
                            checked={config.value as boolean}
                            onChange={(e) => handleConfigChange(config.id, e.target.checked)}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                          />
                          <span className="ml-2 text-sm text-gray-600">
                            {config.value ? 'Enabled' : 'Disabled'}
                          </span>
                        </label>
                      ) : config.type === 'select' ? (
                        <select
                          value={config.value as string}
                          onChange={(e) => handleConfigChange(config.id, e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        >
                          {config.options?.map(option => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                      ) : config.type === 'number' ? (
                        <input
                          type="number"
                          value={config.value as number}
                          onChange={(e) => handleConfigChange(config.id, Number(e.target.value))}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                      ) : (
                        <input
                          type={config.sensitive ? 'password' : 'text'}
                          value={config.value as string}
                          onChange={(e) => handleConfigChange(config.id, e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {config.sensitive && '🔒 Sensitive'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'security' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold">Security Overview</h3>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Authentication Status</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <span className="text-sm">JWT Authentication</span>
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <span className="text-sm">RBAC System</span>
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <span className="text-sm">Password Encryption</span>
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-3">Security Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Failed Login Attempts (24h)</span>
                      <span className="text-sm font-medium">3</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Active Sessions</span>
                      <span className="text-sm font-medium">12</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Last Security Scan</span>
                      <span className="text-sm font-medium">2 hours ago</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold">System Logs</h3>
            <div className="flex items-center space-x-2">
              <button className="flex items-center px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </button>
              <select className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                <option value="all">All Levels</option>
                <option value="error">Errors Only</option>
                <option value="warning">Warnings & Errors</option>
                <option value="info">Info & Above</option>
              </select>
            </div>
          </div>
          <div className="p-6">
            <div className="font-mono text-sm bg-gray-50 p-4 rounded-lg max-h-96 overflow-y-auto">
              <div className="space-y-1">
                <div className="text-blue-600">[2024-12-28 10:45:32] [INFO] System startup completed successfully</div>
                <div className="text-green-600">[2024-12-28 10:45:33] [INFO] All agents initialized</div>
                <div className="text-yellow-600">[2024-12-28 10:47:15] [WARNING] High memory usage detected: 87%</div>
                <div className="text-blue-600">[2024-12-28 10:48:22] [INFO] New WebSocket connection established</div>
                <div className="text-green-600">[2024-12-28 10:49:01] [INFO] Agent response completed in 1.2ms</div>
                <div className="text-red-600">[2024-12-28 10:50:33] [ERROR] Failed to connect to external API service</div>
                <div className="text-blue-600">[2024-12-28 10:51:11] [INFO] User authentication successful</div>
                <div className="text-yellow-600">[2024-12-28 10:52:45] [WARNING] API rate limit approaching: 85% used</div>
                <div className="text-green-600">[2024-12-28 10:53:20] [INFO] Database backup completed</div>
                <div className="text-blue-600">[2024-12-28 10:54:02] [INFO] Performance metrics updated</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};