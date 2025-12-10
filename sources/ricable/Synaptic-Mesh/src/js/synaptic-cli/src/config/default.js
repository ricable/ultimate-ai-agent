export function createDefaultConfig() {
  return {
    project: {
      name: 'synaptic-project',
      template: 'basic',
      version: '1.0.0-alpha.1',
      created: new Date().toISOString()
    },
    mesh: {
      topology: 'mesh',
      defaultAgents: 5,
      coordinationPort: 7070,
      heartbeatInterval: 30000,
      nodeTimeout: 60000
    },
    neural: {
      enabled: true,
      port: 7071,
      defaultModel: 'mlp',
      trainingEnabled: true,
      gpuAcceleration: false
    },
    dag: {
      enabled: true,
      port: 7072,
      maxConcurrentWorkflows: 10,
      workflowTimeout: 300000
    },
    peer: {
      enabled: true,
      port: 7073,
      autoDiscovery: true,
      maxPeers: 50,
      discoveryInterval: 60000,
      protocols: ['libp2p', 'websocket']
    },
    features: {
      mcp: true,
      mcpPort: 3000,
      webui: false,
      webuiPort: 8080,
      monitoring: false,
      logging: true,
      backup: false
    },
    security: {
      encryption: true,
      authentication: false,
      certificates: {
        autoGenerate: true,
        keySize: 2048
      }
    },
    storage: {
      provider: 'sqlite',
      path: '.synaptic/data',
      backup: {
        enabled: false,
        interval: 3600000, // 1 hour
        retention: 7 // days
      }
    },
    logging: {
      level: 'info',
      file: '.synaptic/logs/synaptic.log',
      maxSize: '10MB',
      maxFiles: 5,
      console: true
    },
    performance: {
      workerThreads: 4,
      maxMemory: 1024, // MB
      cacheSize: 100, // MB
      enableOptimizations: true
    }
  };
}