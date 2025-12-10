// Advanced user management interface with RBAC controls
import React, { useState, useEffect } from 'react';
import { Users, UserPlus, Edit, Trash2, Shield, Key, Clock, Search, Filter, Download, Upload } from 'lucide-react';
import { useAuthenticatedApi } from '../../auth/AuthContext';

interface Permission {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  user_count: number;
}

interface User {
  id: string;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  full_name?: string;
  role_id: string;
  roles: string[];
  status: 'active' | 'inactive' | 'locked' | 'pending';
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login: string;
  login_count: number;
  avatar?: string;
  phone?: string;
  department?: string;
  api_key_enabled: boolean;
  two_factor_enabled: boolean;
}

interface UserActivity {
  id: string;
  user_id: string;
  action: string;
  timestamp: string;
  ip_address: string;
  user_agent: string;
  details?: string;
}

export const UserManagement: React.FC = () => {
  const { makeRequest } = useAuthenticatedApi();
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [activities, setActivities] = useState<UserActivity[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [roleFilter, setRoleFilter] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch permissions from API
  const fetchPermissions = async () => {
    try {
      const response = await fetch('/api/admin/permissions');
      if (response.ok) {
        const data = await response.json();
        setPermissions(data.permissions || []);
      } else {
        // Fallback permissions if API not available
        const defaultPermissions: Permission[] = [
          { id: '1', name: 'system_admin', description: 'Full system administration access', category: 'System' },
          { id: '2', name: 'user_management', description: 'Create, edit, and delete users', category: 'Users' },
          { id: '3', name: 'config_management', description: 'Modify system configuration', category: 'Configuration' },
          { id: '4', name: 'view_logs', description: 'Access system logs and audit trails', category: 'Monitoring' },
          { id: '5', name: 'view_analytics', description: 'Access analytics and reports', category: 'Analytics' },
          { id: '6', name: 'manage_agents', description: 'Configure and manage AI agents', category: 'Agents' },
          { id: '7', name: 'use_agents', description: 'Interact with AI agents', category: 'Agents' },
          { id: '8', name: 'view_own_data', description: 'View own user data and activity', category: 'Personal' },
          { id: '9', name: 'view_public', description: 'Access public information only', category: 'Public' },
          { id: '10', name: 'manage_documents', description: 'Upload and manage documents', category: 'Documents' }
        ];
        setPermissions(defaultPermissions);
      }
    } catch (error) {
      console.error('Failed to fetch permissions:', error);
      setPermissions([]);
    }
  };

  // Fetch roles from API
  const fetchRoles = async () => {
    try {
      const rolesData = await makeRequest<any>('/api/auth/roles');
      // Convert backend roles format to frontend format
      const rolesArray: Role[] = Object.entries(rolesData).map(([key, role]: [string, any], index) => ({
        id: (index + 1).toString(),
        name: role.name,
        description: role.description || 'No description',
        permissions: role.permissions || [],
        user_count: 0 // Will be calculated from users
      }));
      setRoles(rolesArray);
    } catch (error) {
      console.error('Failed to fetch roles:', error);
      // Fallback roles if API not available
      const defaultRoles: Role[] = [
        {
          id: '1',
          name: 'admin',
          description: 'System administrators with full access',
          permissions: ['user:create', 'user:read', 'user:update', 'user:delete', 'agent:create', 'agent:read', 'agent:update', 'agent:delete', 'system:admin', 'system:manage', 'system:read', 'websocket:connect'],
          user_count: 1
        },
        {
          id: '2',
          name: 'manager',
          description: 'Team managers with user and agent management access',
          permissions: ['user:read', 'user:update', 'agent:create', 'agent:read', 'agent:update', 'system:manage', 'websocket:connect'],
          user_count: 0
        },
        {
          id: '3',
          name: 'user',
          description: 'Regular users with agent interaction access',
          permissions: ['agent:read', 'agent:create', 'websocket:connect'],
          user_count: 0
        },
        {
          id: '4',
          name: 'guest',
          description: 'Limited access for temporary users',
          permissions: ['agent:read'],
          user_count: 0
        }
      ];
      setRoles(defaultRoles);
    }
  };

  // Fetch users from API
  const fetchUsers = async () => {
    try {
      const data = await makeRequest<{users: any[]}>('/api/users');
      // Map backend user format to frontend format
      const mappedUsers: User[] = (data.users || []).map((backendUser: any) => ({
        id: backendUser.id,
        username: backendUser.username,
        email: backendUser.email,
        first_name: backendUser.full_name?.split(' ')[0] || backendUser.username,
        last_name: backendUser.full_name?.split(' ').slice(1).join(' ') || '',
        full_name: backendUser.full_name,
        role_id: backendUser.roles?.[0] || 'user', // Use first role as primary role
        roles: backendUser.roles || ['user'],
        status: backendUser.is_active ? 'active' : 'inactive',
        is_active: backendUser.is_active,
        is_verified: backendUser.is_verified,
        created_at: backendUser.created_at,
        last_login: backendUser.last_login || new Date().toISOString(),
        login_count: 1, // Default value since backend doesn't track this
        api_key_enabled: false, // Default value
        two_factor_enabled: false, // Default value
        department: 'Engineering' // Default value
      }));
      setUsers(mappedUsers);
    } catch (error) {
      console.error('Failed to fetch users:', error);
      setUsers([]);
    }
  };

  // Fetch user activities from API
  const fetchActivities = async () => {
    try {
      const response = await fetch('/api/admin/activities');
      if (response.ok) {
        const data = await response.json();
        setActivities(data.activities || []);
      } else {
        console.error('Failed to fetch activities:', response.statusText);
        setActivities([]);
      }
    } catch (error) {
      console.error('Failed to fetch activities:', error);
      setActivities([]);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await Promise.all([
        fetchUsers(),
        fetchRoles(),
        fetchPermissions(),
        fetchActivities()
      ]);
      setIsLoading(false);
    };
    
    loadData();
  }, []);

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         `${user.first_name} ${user.last_name}`.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || user.status === statusFilter;
    const matchesRole = roleFilter === 'all' || user.role_id === roleFilter;
    
    return matchesSearch && matchesStatus && matchesRole;
  });

  const getRoleById = (roleId: string) => roles.find(r => r.id === roleId);
  const getPermissionById = (permissionId: string) => permissions.find(p => p.id === permissionId);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'inactive': return 'bg-gray-100 text-gray-800';
      case 'locked': return 'bg-red-100 text-red-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRoleColor = (roleName: string) => {
    switch (roleName) {
      case 'admin': return 'bg-red-100 text-red-800';
      case 'manager': return 'bg-blue-100 text-blue-800';
      case 'user': return 'bg-green-100 text-green-800';
      case 'guest': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleStatusChange = async (userId: string, newStatus: string) => {
    try {
      await makeRequest(`/api/users/${userId}`, {
        method: 'PUT',
        body: {
          is_active: newStatus === 'active'
        }
      });
      setUsers(prev => prev.map(user => 
        user.id === userId ? { ...user, status: newStatus as any, is_active: newStatus === 'active' } : user
      ));
    } catch (error) {
      console.error('Failed to update user status:', error);
      alert('Failed to update user status. Please try again.');
    }
  };

  const handleDeleteUser = async (userId: string) => {
    if (confirm('Are you sure you want to delete this user?')) {
      try {
        await makeRequest(`/api/users/${userId}`, { method: 'DELETE' });
        setUsers(prev => prev.filter(user => user.id !== userId));
      } catch (error) {
        console.error('Failed to delete user:', error);
        alert('Failed to delete user. Please try again.');
      }
    }
  };

  const exportUsers = () => {
    const csvData = [
      ['Username', 'Email', 'Name', 'Role', 'Status', 'Created', 'Last Login', 'Login Count'],
      ...filteredUsers.map(user => [
        user.username,
        user.email,
        `${user.first_name} ${user.last_name}`,
        getRoleById(user.role_id)?.name || 'Unknown',
        user.status,
        new Date(user.created_at).toLocaleDateString(),
        new Date(user.last_login).toLocaleDateString(),
        user.login_count.toString()
      ])
    ];
    
    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `uap-users-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
            ))}
          </div>
          <div className="bg-gray-200 rounded-lg h-96"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">User Management</h1>
        <div className="flex items-center space-x-2">
          <button
            onClick={exportUsers}
            className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <UserPlus className="h-4 w-4 mr-2" />
            Add User
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Users</p>
              <p className="text-2xl font-bold text-gray-900">{users.length}</p>
            </div>
            <Users className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Users</p>
              <p className="text-2xl font-bold text-green-600">{users.filter(u => u.status === 'active').length}</p>
            </div>
            <Shield className="h-8 w-8 text-green-500" />
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Pending Approval</p>
              <p className="text-2xl font-bold text-yellow-600">{users.filter(u => u.status === 'pending').length}</p>
            </div>
            <Clock className="h-8 w-8 text-yellow-500" />
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">API Access</p>
              <p className="text-2xl font-bold text-purple-600">{users.filter(u => u.api_key_enabled).length}</p>
            </div>
            <Key className="h-8 w-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <input
              type="text"
              placeholder="Search users..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="locked">Locked</option>
            <option value="pending">Pending</option>
          </select>
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Roles</option>
            {roles.map(role => (
              <option key={role.id} value={role.id}>{role.name}</option>
            ))}
          </select>
          <div className="text-sm text-gray-500 flex items-center">
            <Filter className="h-4 w-4 mr-2" />
            {filteredUsers.length} of {users.length} users
          </div>
        </div>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Department</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Login</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Security</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredUsers.map((user) => {
                const role = getRoleById(user.role_id);
                return (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="h-10 w-10 bg-gray-300 rounded-full flex items-center justify-center">
                          <span className="text-sm font-medium text-gray-700">
                            {user.first_name.charAt(0)}{user.last_name.charAt(0)}
                          </span>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {user.first_name} {user.last_name}
                          </div>
                          <div className="text-sm text-gray-500">{user.email}</div>
                          <div className="text-xs text-gray-400">@{user.username}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getRoleColor(role?.name || '')}`}>
                        {role?.name || 'Unknown'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <select
                        value={user.status}
                        onChange={(e) => handleStatusChange(user.id, e.target.value)}
                        className={`text-xs font-semibold rounded-full px-2 py-1 border-0 ${getStatusColor(user.status)}`}
                      >
                        <option value="active">Active</option>
                        <option value="inactive">Inactive</option>
                        <option value="locked">Locked</option>
                        <option value="pending">Pending</option>
                      </select>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {user.department || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <div>{new Date(user.last_login).toLocaleDateString()}</div>
                      <div className="text-xs text-gray-400">
                        {user.login_count} logins
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {user.api_key_enabled && (
                          <Key className="h-4 w-4 text-purple-500" title="API Access" />
                        )}
                        {user.two_factor_enabled && (
                          <Shield className="h-4 w-4 text-green-500" title="2FA Enabled" />
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => {
                            setSelectedUser(user);
                            setShowEditModal(true);
                          }}
                          className="text-blue-600 hover:text-blue-900"
                        >
                          <Edit className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteUser(user.id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent User Activity */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">Recent User Activity</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {activities.slice(0, 5).map((activity) => {
              const user = users.find(u => u.id === activity.user_id);
              return (
                <div key={activity.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="text-sm font-medium">
                      {user?.username || 'Unknown User'} - {activity.action}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(activity.timestamp).toLocaleString()} from {activity.ip_address}
                    </div>
                    {activity.details && (
                      <div className="text-xs text-gray-400 mt-1">{activity.details}</div>
                    )}
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    activity.action.includes('Failed') ? 'bg-red-100 text-red-800' :
                    activity.action.includes('Login') ? 'bg-green-100 text-green-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {activity.action}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};