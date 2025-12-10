// frontend/src/components/robotics/RoboticsDashboard.tsx
// Agent 34: Advanced Robotics Integration - Frontend Dashboard

import React, { useState, useEffect } from 'react';

interface RoboticsStatus {
  system_running: boolean;
  navigation_status: any;
  sensor_status: any;
  safety_status: any;
  robot_coordinator_status: any;
  timestamp: string;
}

interface HumanCommand {
  command_id: string;
  intent: string;
  entities: any;
  confidence: number;
}

interface RobotResponse {
  response_id: string;
  message: string;
  response_type: string;
  action_taken?: string;
  additional_data?: any;
}

interface InteractionResult {
  command: HumanCommand;
  response: RobotResponse;
}

const RoboticsDashboard: React.FC = () => {
  const [status, setStatus] = useState<RoboticsStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState('');
  const [interactions, setInteractions] = useState<InteractionResult[]>([]);
  const [interactionStats, setInteractionStats] = useState<any>(null);

  // Fetch robotics status
  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/robotics/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch (error) {
      console.error('Failed to fetch robotics status:', error);
    }
  };

  // Process human command
  const processCommand = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/api/robotics/interaction/process-command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_text: inputText,
          interaction_mode: 'text_command'
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setInteractions(prev => [result, ...prev.slice(0, 9)]); // Keep last 10
        setInputText('');
        
        // Refresh stats
        fetchInteractionStats();
      }
    } catch (error) {
      console.error('Failed to process command:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch interaction statistics
  const fetchInteractionStats = async () => {
    try {
      const response = await fetch('/api/robotics/interaction/stats');
      if (response.ok) {
        const data = await response.json();
        setInteractionStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch interaction stats:', error);
    }
  };

  // Navigation commands
  const sendNavigationCommand = async (x: number, y: number) => {
    try {
      const response = await fetch('/api/robotics/navigation/set-goal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_x: x,
          target_y: y,
          target_z: 0,
          priority: 1,
          tolerance: 0.2
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Navigation goal set:', result);
        fetchStatus(); // Refresh status
      }
    } catch (error) {
      console.error('Failed to set navigation goal:', error);
    }
  };

  // Safety commands
  const triggerEmergencyStop = async () => {
    try {
      const response = await fetch('/api/robotics/actuators/emergency-stop', {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Emergency stop triggered:', result);
        fetchStatus();
      }
    } catch (error) {
      console.error('Failed to trigger emergency stop:', error);
    }
  };

  // Demo commands
  const demoCommands = [
    "Go to position 2, 3",
    "Pick up the bottle",
    "What is your status?",
    "Navigate to the kitchen",
    "Stop immediately",
    "Start inspection task",
    "Move to coordinates 1, 1",
    "How are you doing?",
    "Deliver the package",
    "Clean the area"
  ];

  useEffect(() => {
    fetchStatus();
    fetchInteractionStats();
    
    // Refresh status every 5 seconds
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'safe':
      case 'idle':
      case 'completed':
        return 'text-green-600';
      case 'active':
      case 'navigating':
      case 'in_progress':
        return 'text-blue-600';
      case 'warning':
      case 'caution':
        return 'text-yellow-600';
      case 'error':
      case 'emergency':
      case 'unsafe':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Advanced Robotics Integration
        </h1>
        <p className="text-gray-600">
          Unified robotic process automation, navigation, and human-robot interaction
        </p>
      </div>

      {/* System Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">System Status</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>System:</span>
              <span className={status?.system_running ? 'text-green-600' : 'text-red-600'}>
                {status?.system_running ? 'Running' : 'Stopped'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Navigation:</span>
              <span className={getStatusColor(status?.navigation_status?.status)}>
                {status?.navigation_status?.status || 'Unknown'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Safety:</span>
              <span className={getStatusColor(status?.safety_status?.safety_state)}>
                {status?.safety_status?.safety_state || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Robot Coordination</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Robots:</span>
              <span>{status?.robot_coordinator_status?.total_robots || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Tasks:</span>
              <span>{status?.robot_coordinator_status?.pending_tasks || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Active:</span>
              <span className="text-blue-600">
                {status?.robot_coordinator_status?.robot_status?.active || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Safety Monitor</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Alerts:</span>
              <span className={status?.safety_status?.active_alerts > 0 ? 'text-yellow-600' : 'text-green-600'}>
                {status?.safety_status?.active_alerts || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span>E-Stops:</span>
              <span className={status?.safety_status?.emergency_stop_active ? 'text-red-600' : 'text-green-600'}>
                {status?.safety_status?.emergency_stop_active ? 'Active' : 'None'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Uptime:</span>
              <span>{status?.safety_status?.uptime_hours?.toFixed(1) || 0}h</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Interaction Stats</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Commands:</span>
              <span>{interactionStats?.total_interactions || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Success:</span>
              <span className="text-green-600">
                {interactionStats?.successful_commands || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Success Rate:</span>
              <span>{((interactionStats?.success_rate || 0) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Human-Robot Interaction */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Human-Robot Interface</h3>
          
          <div className="mb-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && processCommand()}
                placeholder="Type a command (e.g., 'Go to position 2, 3' or 'Pick up the bottle')"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <button
                onClick={processCommand}
                disabled={loading || !inputText.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Send'}
              </button>
            </div>
          </div>

          <div className="mb-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Demo Commands:</h4>
            <div className="flex flex-wrap gap-2">
              {demoCommands.slice(0, 6).map((cmd, idx) => (
                <button
                  key={idx}
                  onClick={() => setInputText(cmd)}
                  className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-md"
                >
                  {cmd}
                </button>
              ))}
            </div>
          </div>

          {/* Recent Interactions */}
          <div className="max-h-96 overflow-y-auto">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Recent Interactions:</h4>
            {interactions.length === 0 ? (
              <p className="text-gray-500 text-sm">No interactions yet. Try sending a command!</p>
            ) : (
              <div className="space-y-3">
                {interactions.map((interaction, idx) => (
                  <div key={idx} className="border-l-4 border-blue-200 pl-4 pb-3">
                    <div className="bg-blue-50 p-2 rounded mb-2">
                      <div className="text-sm font-medium text-blue-800">
                        You: {interaction.command.intent} (confidence: {(interaction.command.confidence * 100).toFixed(0)}%)
                      </div>
                    </div>
                    <div className="bg-gray-50 p-2 rounded">
                      <div className="text-sm text-gray-800">
                        Robot: {interaction.response.message}
                      </div>
                      {interaction.response.action_taken && (
                        <div className="text-xs text-gray-600 mt-1">
                          Action: {interaction.response.action_taken}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quick Controls */}
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Quick Controls</h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Navigation</h4>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => sendNavigationCommand(-1, 1)}
                  className="px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded text-sm"
                >
                  Kitchen (-1,1)
                </button>
                <button
                  onClick={() => sendNavigationCommand(0, 0)}
                  className="px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded text-sm"
                >
                  Home (0,0)
                </button>
                <button
                  onClick={() => sendNavigationCommand(2, 1)}
                  className="px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded text-sm"
                >
                  Office (2,1)
                </button>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Safety</h4>
              <button
                onClick={triggerEmergencyStop}
                className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium"
              >
                🛑 Emergency Stop
              </button>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">System Control</h4>
              <div className="grid grid-cols-2 gap-2">
                <button className="px-3 py-2 bg-green-100 hover:bg-green-200 rounded text-sm">
                  Start Systems
                </button>
                <button className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm">
                  Refresh Status
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Component Information */}
      <div className="bg-white rounded-lg border p-6">
        <h3 className="text-lg font-semibold mb-4">Robotics Components</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">RPA Engine</h4>
            <p className="text-gray-600">Robotic process automation with computer vision integration</p>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">Navigation System</h4>
            <p className="text-gray-600">A* path planning with dynamic obstacle avoidance</p>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">Sensor Fusion</h4>
            <p className="text-gray-600">Multi-sensor data fusion with Kalman filtering</p>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">Vision Processing</h4>
            <p className="text-gray-600">Computer vision for obstacle detection and object recognition</p>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">Multi-Robot Coordination</h4>
            <p className="text-gray-600">Task allocation and conflict resolution between robots</p>
          </div>
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-900">Safety Monitoring</h4>
            <p className="text-gray-600">Real-time safety monitoring with collision detection</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoboticsDashboard;