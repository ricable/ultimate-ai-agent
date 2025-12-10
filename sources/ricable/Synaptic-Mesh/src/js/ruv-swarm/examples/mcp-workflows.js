/**
 * Comprehensive MCP Workflow Examples for ruv-swarm
 * Demonstrates all MCP tool capabilities with real-world scenarios
 */

import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';

// MCP Client wrapper for easier use
class RuvSwarmMCP {
    constructor(url = 'ws://localhost:3000/mcp') {
        this.url = url;
        this.ws = null;
        this.requestId = 0;
        this.pendingRequests = new Map();
        this.eventHandlers = new Map();
    }

    async connect(clientInfo = { name: 'ruv-swarm-client', version: '1.0.0' }) {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            
            this.ws.on('open', async () => {
                try {
                    const result = await this.call('initialize', { clientInfo });
                    console.log(`✅ Connected to RUV-Swarm MCP Server v${result.serverInfo.version}`);
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            });
            
            this.ws.on('message', (data) => {
                const message = JSON.parse(data.toString());
                
                if (message.id && this.pendingRequests.has(message.id)) {
                    const { resolve, reject } = this.pendingRequests.get(message.id);
                    this.pendingRequests.delete(message.id);
                    
                    if (message.error) {
                        reject(new Error(message.error.message));
                    } else {
                        resolve(message.result);
                    }
                } else if (message.method) {
                    // Handle notifications
                    this.handleNotification(message);
                }
            });
            
            this.ws.on('error', reject);
        });
    }

    async disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    async call(method, params = null) {
        const id = ++this.requestId;
        const request = { jsonrpc: '2.0', id, method, params };

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });
            this.ws.send(JSON.stringify(request));
            
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error(`Request ${id} timed out`));
                }
            }, 30000);
        });
    }

    async tool(name, args) {
        return this.call('tools/call', { name, arguments: args });
    }

    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    handleNotification(message) {
        const handlers = this.eventHandlers.get(message.method) || [];
        handlers.forEach(handler => handler(message.params));
    }

    // Helper methods for common operations
    async spawnAgent(type, name = null, capabilities = {}) {
        return this.tool('ruv-swarm.spawn', {
            agent_type: type,
            name,
            capabilities
        });
    }

    async orchestrate(objective, options = {}) {
        return this.tool('ruv-swarm.orchestrate', {
            objective,
            strategy: options.strategy || 'development',
            mode: options.mode || 'hierarchical',
            max_agents: options.maxAgents || 5,
            parallel: options.parallel !== false
        });
    }

    async storeMemory(key, value, ttl = null) {
        return this.tool('ruv-swarm.memory.store', {
            key,
            value,
            ttl_secs: ttl
        });
    }

    async getMemory(key) {
        return this.tool('ruv-swarm.memory.get', { key });
    }

    async createTask(type, description, priority = 'medium', assignedAgent = null) {
        return this.tool('ruv-swarm.task.create', {
            task_type: type,
            description,
            priority,
            assigned_agent: assignedAgent
        });
    }

    async query(filter = {}, includeMetrics = false) {
        return this.tool('ruv-swarm.query', {
            filter,
            include_metrics: includeMetrics
        });
    }

    async monitor(eventTypes = ['all'], duration = 60) {
        return this.tool('ruv-swarm.monitor', {
            event_types: eventTypes,
            duration_secs: duration
        });
    }

    async optimize(targetMetric = 'throughput', constraints = {}, autoApply = false) {
        return this.tool('ruv-swarm.optimize', {
            target_metric: targetMetric,
            constraints,
            auto_apply: autoApply
        });
    }

    async executeWorkflow(path, parameters = {}, async = false) {
        return this.tool('ruv-swarm.workflow.execute', {
            workflow_path: path,
            parameters,
            async_execution: async
        });
    }

    async listAgents(includeInactive = false, sortBy = 'created_at') {
        return this.tool('ruv-swarm.agent.list', {
            include_inactive: includeInactive,
            sort_by: sortBy
        });
    }
}

// Example Workflows

/**
 * Workflow 1: Complete Web Application Development
 * Demonstrates full development lifecycle with multiple agent types
 */
async function webAppDevelopmentWorkflow(client) {
    console.log('\n🚀 Workflow 1: Complete Web Application Development\n');
    
    // Phase 1: Research and Planning
    console.log('📋 Phase 1: Research and Planning');
    
    // Spawn research team
    const researchLead = await client.spawnAgent('researcher', 'research-lead', {
        specialization: 'web_architecture',
        max_concurrent_tasks: 3
    });
    console.log(`  ✅ Spawned Research Lead: ${researchLead.agent_id}`);
    
    // Store project requirements
    await client.storeMemory('project_requirements', {
        name: 'E-Commerce Platform',
        features: [
            'User authentication',
            'Product catalog',
            'Shopping cart',
            'Payment integration',
            'Order management'
        ],
        tech_stack: {
            frontend: 'React + TypeScript',
            backend: 'Node.js + Express',
            database: 'PostgreSQL',
            cache: 'Redis'
        },
        constraints: {
            timeline: '3 months',
            budget: '$50,000',
            team_size: 5
        }
    });
    console.log('  ✅ Stored project requirements');
    
    // Create research tasks
    const researchTask = await client.createTask(
        'research',
        'Analyze best practices for e-commerce platform architecture',
        'high',
        researchLead.agent_id
    );
    console.log(`  ✅ Created research task: ${researchTask.task_id}`);
    
    // Phase 2: Architecture Design
    console.log('\n🏗️  Phase 2: Architecture Design');
    
    // Orchestrate architecture design
    const architectureResult = await client.orchestrate(
        'Design scalable microservices architecture for e-commerce platform',
        {
            strategy: 'development',
            mode: 'centralized',
            maxAgents: 3
        }
    );
    console.log(`  ✅ Architecture orchestration started: ${architectureResult.task_id}`);
    
    // Store architecture decisions
    await client.storeMemory('architecture_decisions', {
        pattern: 'microservices',
        services: [
            { name: 'auth-service', responsibility: 'User authentication and authorization' },
            { name: 'product-service', responsibility: 'Product catalog management' },
            { name: 'cart-service', responsibility: 'Shopping cart operations' },
            { name: 'order-service', responsibility: 'Order processing and management' },
            { name: 'payment-service', responsibility: 'Payment processing' }
        ],
        communication: 'REST API with message queue for async operations',
        deployment: 'Kubernetes with auto-scaling'
    });
    
    // Phase 3: Development Team Assembly
    console.log('\n👥 Phase 3: Development Team Assembly');
    
    const team = {
        frontend: [],
        backend: [],
        testers: []
    };
    
    // Spawn frontend developers
    for (let i = 0; i < 2; i++) {
        const dev = await client.spawnAgent('coder', `frontend-dev-${i}`, {
            specialization: 'react',
            skills: ['typescript', 'redux', 'css-in-js']
        });
        team.frontend.push(dev.agent_id);
    }
    console.log(`  ✅ Spawned ${team.frontend.length} frontend developers`);
    
    // Spawn backend developers
    for (let i = 0; i < 3; i++) {
        const dev = await client.spawnAgent('coder', `backend-dev-${i}`, {
            specialization: 'nodejs',
            skills: ['express', 'postgresql', 'redis', 'microservices']
        });
        team.backend.push(dev.agent_id);
    }
    console.log(`  ✅ Spawned ${team.backend.length} backend developers`);
    
    // Spawn testers
    for (let i = 0; i < 2; i++) {
        const tester = await client.spawnAgent('tester', `qa-engineer-${i}`, {
            specialization: i === 0 ? 'unit_testing' : 'integration_testing',
            tools: ['jest', 'cypress', 'postman']
        });
        team.testers.push(tester.agent_id);
    }
    console.log(`  ✅ Spawned ${team.testers.length} QA engineers`);
    
    // Store team structure
    await client.storeMemory('development_team', team);
    
    // Phase 4: Sprint Planning and Task Distribution
    console.log('\n📊 Phase 4: Sprint Planning and Task Distribution');
    
    // Create development tasks for Sprint 1
    const sprint1Tasks = [
        { type: 'development', desc: 'Implement auth-service with JWT', assignee: team.backend[0] },
        { type: 'development', desc: 'Create user registration UI', assignee: team.frontend[0] },
        { type: 'development', desc: 'Setup PostgreSQL schemas', assignee: team.backend[1] },
        { type: 'testing', desc: 'Write auth-service unit tests', assignee: team.testers[0] },
        { type: 'development', desc: 'Implement product-service CRUD', assignee: team.backend[2] },
        { type: 'development', desc: 'Create product listing components', assignee: team.frontend[1] }
    ];
    
    const taskIds = [];
    for (const task of sprint1Tasks) {
        const result = await client.createTask(
            task.type,
            task.desc,
            'high',
            task.assignee
        );
        taskIds.push(result.task_id);
    }
    console.log(`  ✅ Created ${taskIds.length} tasks for Sprint 1`);
    
    // Phase 5: Development Monitoring
    console.log('\n📡 Phase 5: Development Monitoring');
    
    // Set up event monitoring
    client.on('ruv-swarm/event', (event) => {
        console.log(`  📢 Event: ${event.event.type} - ${event.event.data.message || ''}`);
    });
    
    // Start monitoring
    await client.monitor(['task_started', 'task_completed', 'agent_message'], 10);
    console.log('  ✅ Monitoring active for 10 seconds');
    
    // Phase 6: Performance Optimization
    console.log('\n⚡ Phase 6: Performance Optimization');
    
    const optimization = await client.optimize('throughput', {
        max_memory_mb: 1024,
        max_cpu_percent: 75,
        min_agents: 5
    }, false);
    
    console.log('  ✅ Optimization recommendations:');
    optimization.recommendations.forEach(rec => {
        console.log(`    - ${rec}`);
    });
    
    // Phase 7: Status Review
    console.log('\n📈 Phase 7: Status Review');
    
    const status = await client.query({}, true);
    console.log(`  Total Agents: ${status.total_agents}`);
    console.log(`  Active Tasks: ${status.active_tasks}`);
    console.log(`  Completed Tasks: ${status.completed_tasks}`);
    
    if (status.metrics) {
        console.log('  Performance Metrics:');
        console.log(`    - CPU Usage: ${status.metrics.cpu_usage || 'N/A'}%`);
        console.log(`    - Memory Usage: ${status.metrics.memory_usage || 'N/A'}MB`);
        console.log(`    - Task Throughput: ${status.metrics.task_throughput || 'N/A'}/min`);
    }
}

/**
 * Workflow 2: AI Research Swarm
 * Demonstrates distributed research with neural learning
 */
async function aiResearchSwarmWorkflow(client) {
    console.log('\n🧠 Workflow 2: AI Research Swarm\n');
    
    // Initialize research context
    await client.storeMemory('research_context', {
        topic: 'Large Language Model Optimization Techniques',
        subtopics: [
            'Model compression and quantization',
            'Knowledge distillation',
            'Efficient attention mechanisms',
            'Parameter-efficient fine-tuning',
            'Inference optimization'
        ],
        output_format: 'comprehensive_report',
        deadline: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
    });
    
    // Orchestrate research swarm
    const research = await client.orchestrate(
        'Conduct comprehensive research on LLM optimization techniques',
        {
            strategy: 'research',
            mode: 'mesh',  // Mesh network for collaborative research
            maxAgents: 8,
            parallel: true
        }
    );
    console.log(`  ✅ Research swarm initiated: ${research.task_id}`);
    
    // Create specialized research agents
    const researchers = [];
    const specializations = [
        'model_compression',
        'attention_mechanisms',
        'training_efficiency',
        'hardware_optimization'
    ];
    
    for (const spec of specializations) {
        const agent = await client.spawnAgent('researcher', `${spec}-expert`, {
            specialization: spec,
            neural_enabled: true,
            learning_rate: 0.01,
            knowledge_base: 'academic_papers'
        });
        researchers.push(agent);
        console.log(`  ✅ Spawned ${spec} expert: ${agent.agent_id}`);
    }
    
    // Create analysis tasks
    for (let i = 0; i < researchers.length; i++) {
        await client.createTask(
            'research',
            `Deep dive into ${specializations[i]} techniques and innovations`,
            'critical',
            researchers[i].agent_id
        );
    }
    
    // Store intermediate findings
    await client.storeMemory('research_findings_compression', {
        techniques: {
            quantization: {
                methods: ['INT8', 'INT4', 'Mixed precision'],
                performance_impact: '2-4x speedup with <1% accuracy loss',
                best_practices: ['Post-training quantization', 'Quantization-aware training']
            },
            pruning: {
                methods: ['Magnitude pruning', 'Structured pruning', 'Movement pruning'],
                sparsity_levels: '50-90% achievable',
                hardware_support: ['NVIDIA Ampere', 'Intel VNNI']
            }
        },
        timestamp: new Date().toISOString()
    });
    
    // Neural network pattern learning simulation
    console.log('\n🔬 Neural Pattern Learning');
    
    const analyst = await client.spawnAgent('analyst', 'neural-pattern-analyzer', {
        neural_enabled: true,
        pattern_recognition: true,
        learning_epochs: 1000
    });
    
    // Store pattern data for learning
    await client.storeMemory('optimization_patterns', {
        patterns: [
            { technique: 'quantization', model_size: 'large', speedup: 3.2 },
            { technique: 'pruning', model_size: 'large', speedup: 2.8 },
            { technique: 'distillation', model_size: 'medium', speedup: 1.5 },
            { technique: 'quantization', model_size: 'medium', speedup: 2.1 }
        ],
        target: 'predict optimal technique based on model characteristics'
    });
    
    await client.createTask(
        'analysis',
        'Learn optimization patterns and predict best techniques',
        'high',
        analyst.agent_id
    );
    
    console.log(`  ✅ Neural learning task created for pattern analysis`);
}

/**
 * Workflow 3: Continuous Integration/Deployment Pipeline
 * Demonstrates maintenance and optimization workflows
 */
async function cicdPipelineWorkflow(client) {
    console.log('\n🔄 Workflow 3: CI/CD Pipeline Automation\n');
    
    // Setup pipeline configuration
    await client.storeMemory('pipeline_config', {
        stages: ['build', 'test', 'security_scan', 'deploy'],
        environments: ['dev', 'staging', 'production'],
        triggers: {
            push: 'feature/*',
            pull_request: 'main',
            schedule: '0 2 * * *'  // Daily at 2 AM
        },
        notifications: {
            slack: '#deployments',
            email: 'devops@company.com'
        }
    });
    
    // Spawn CI/CD agents
    const cicdAgents = {
        builder: await client.spawnAgent('coder', 'build-agent', {
            tools: ['docker', 'npm', 'webpack'],
            parallel_builds: 4
        }),
        tester: await client.spawnAgent('tester', 'test-runner', {
            frameworks: ['jest', 'cypress', 'playwright'],
            coverage_threshold: 80
        }),
        security: await client.spawnAgent('analyst', 'security-scanner', {
            tools: ['snyk', 'sonarqube', 'owasp-zap'],
            vulnerability_threshold: 'medium'
        }),
        deployer: await client.spawnAgent('coder', 'deploy-agent', {
            platforms: ['kubernetes', 'aws', 'vercel'],
            rollback_enabled: true
        })
    };
    
    console.log('  ✅ CI/CD agent team assembled');
    
    // Execute pipeline workflow
    const fs = require('fs').promises;
    const pipelineWorkflow = {
        name: 'ci-cd-pipeline',
        version: '1.0.0',
        stages: [
            {
                name: 'build',
                agent: cicdAgents.builder.agent_id,
                tasks: [
                    'Pull latest code from repository',
                    'Install dependencies',
                    'Run build process',
                    'Create Docker image'
                ],
                on_failure: 'abort'
            },
            {
                name: 'test',
                agent: cicdAgents.tester.agent_id,
                tasks: [
                    'Run unit tests',
                    'Run integration tests',
                    'Generate coverage report',
                    'Run E2E tests'
                ],
                parallel: true,
                on_failure: 'abort'
            },
            {
                name: 'security',
                agent: cicdAgents.security.agent_id,
                tasks: [
                    'Scan dependencies for vulnerabilities',
                    'Run static code analysis',
                    'Perform security audit'
                ],
                on_failure: 'notify'
            },
            {
                name: 'deploy',
                agent: cicdAgents.deployer.agent_id,
                tasks: [
                    'Deploy to staging environment',
                    'Run smoke tests',
                    'Deploy to production',
                    'Verify deployment health'
                ],
                requires_approval: true
            }
        ]
    };
    
    const workflowPath = '/tmp/cicd-workflow.json';
    await fs.writeFile(workflowPath, JSON.stringify(pipelineWorkflow, null, 2));
    
    const execution = await client.executeWorkflow(workflowPath, {
        branch: 'feature/user-auth',
        commit: 'abc123def',
        author: 'developer@company.com'
    }, true);
    
    console.log(`  ✅ Pipeline workflow started: ${execution.workflow_id}`);
    
    // Monitor pipeline execution
    await client.monitor(['task_started', 'task_completed', 'workflow_stage_complete'], 15);
    
    // Optimize pipeline performance
    const pipelineOptimization = await client.optimize('latency', {
        max_parallel_tasks: 10,
        cache_dependencies: true,
        incremental_builds: true
    }, true);
    
    console.log('  ✅ Pipeline optimizations applied');
    
    // Clean up
    await fs.unlink(workflowPath);
}

/**
 * Workflow 4: Data Analysis and Machine Learning Pipeline
 * Demonstrates complex data processing with neural networks
 */
async function dataAnalysisPipeline(client) {
    console.log('\n📊 Workflow 4: Data Analysis and ML Pipeline\n');
    
    // Store dataset metadata
    await client.storeMemory('dataset_metadata', {
        name: 'customer_behavior_analysis',
        size: '10GB',
        records: 1000000,
        features: 50,
        target: 'customer_churn',
        split: { train: 0.7, validation: 0.15, test: 0.15 }
    });
    
    // Orchestrate data processing pipeline
    const pipeline = await client.orchestrate(
        'Process customer data and train churn prediction model',
        {
            strategy: 'analysis',
            mode: 'hierarchical',
            maxAgents: 6,
            parallel: true
        }
    );
    
    console.log(`  ✅ Data pipeline orchestration started: ${pipeline.task_id}`);
    
    // Create specialized data agents
    const dataTeam = {
        preprocessor: await client.spawnAgent('analyst', 'data-preprocessor', {
            skills: ['pandas', 'numpy', 'data_cleaning'],
            memory_limit: '8GB'
        }),
        feature_engineer: await client.spawnAgent('analyst', 'feature-engineer', {
            skills: ['feature_extraction', 'dimensionality_reduction'],
            techniques: ['PCA', 'LDA', 'autoencoders']
        }),
        ml_engineer: await client.spawnAgent('analyst', 'ml-engineer', {
            neural_enabled: true,
            frameworks: ['tensorflow', 'pytorch', 'scikit-learn'],
            gpu_enabled: true
        }),
        evaluator: await client.spawnAgent('tester', 'model-evaluator', {
            metrics: ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            visualization: true
        })
    };
    
    // Create pipeline tasks
    const tasks = [
        {
            agent: dataTeam.preprocessor.agent_id,
            task: 'Clean and preprocess customer dataset'
        },
        {
            agent: dataTeam.feature_engineer.agent_id,
            task: 'Engineer features for churn prediction'
        },
        {
            agent: dataTeam.ml_engineer.agent_id,
            task: 'Train neural network for churn prediction'
        },
        {
            agent: dataTeam.evaluator.agent_id,
            task: 'Evaluate model performance and generate report'
        }
    ];
    
    for (const t of tasks) {
        await client.createTask('analysis', t.task, 'high', t.agent);
    }
    
    console.log(`  ✅ Created ${tasks.length} pipeline tasks`);
    
    // Store model training configuration
    await client.storeMemory('model_config', {
        architecture: {
            type: 'feedforward_neural_network',
            layers: [
                { type: 'input', size: 50 },
                { type: 'dense', size: 128, activation: 'relu' },
                { type: 'dropout', rate: 0.3 },
                { type: 'dense', size: 64, activation: 'relu' },
                { type: 'dropout', rate: 0.2 },
                { type: 'dense', size: 1, activation: 'sigmoid' }
            ]
        },
        training: {
            optimizer: 'adam',
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            early_stopping: { patience: 10, monitor: 'val_loss' }
        }
    });
    
    // Simulate training progress
    console.log('\n  📈 Training Progress:');
    const epochs = [1, 10, 25, 50, 75, 100];
    for (const epoch of epochs) {
        await client.storeMemory(`training_metrics_epoch_${epoch}`, {
            epoch,
            train_loss: 0.5 * Math.exp(-epoch / 50),
            val_loss: 0.52 * Math.exp(-epoch / 50),
            train_accuracy: 0.6 + 0.35 * (1 - Math.exp(-epoch / 30)),
            val_accuracy: 0.58 + 0.34 * (1 - Math.exp(-epoch / 30))
        });
        console.log(`    Epoch ${epoch}: Accuracy ${(0.58 + 0.34 * (1 - Math.exp(-epoch / 30))).toFixed(3)}`);
    }
    
    // Store final model results
    await client.storeMemory('model_results', {
        best_epoch: 87,
        metrics: {
            accuracy: 0.924,
            precision: 0.891,
            recall: 0.876,
            f1_score: 0.883,
            auc_roc: 0.956
        },
        feature_importance: {
            'days_since_last_purchase': 0.23,
            'total_purchase_amount': 0.18,
            'support_tickets': 0.15,
            'login_frequency': 0.12,
            'product_views': 0.09
        },
        recommendations: [
            'Deploy model to production with A/B testing',
            'Monitor for data drift weekly',
            'Retrain model monthly with new data',
            'Consider ensemble with gradient boosting'
        ]
    });
    
    console.log('\n  ✅ Model training completed successfully');
}

/**
 * Workflow 5: Multi-Agent Swarm Coordination
 * Demonstrates complex swarm behaviors and coordination patterns
 */
async function swarmCoordinationWorkflow(client) {
    console.log('\n🐝 Workflow 5: Multi-Agent Swarm Coordination\n');
    
    // Define swarm mission
    await client.storeMemory('swarm_mission', {
        objective: 'Build comprehensive documentation system',
        components: [
            'API documentation generator',
            'Code example extractor',
            'Tutorial creator',
            'Interactive playground',
            'Search indexer'
        ],
        coordination_rules: {
            communication: 'broadcast',
            consensus: 'majority_vote',
            task_allocation: 'capability_based',
            conflict_resolution: 'priority_based'
        }
    });
    
    // Create swarm topology
    const swarmConfig = {
        topology: 'mesh',  // Full mesh for maximum coordination
        size: 10,
        redundancy: 2,     // Each task has 2 agents
        heartbeat_interval: 5000
    };
    
    // Initialize swarm with diverse agent types
    const swarm = {
        coordinators: [],
        workers: [],
        reviewers: []
    };
    
    // Spawn coordinator agents
    for (let i = 0; i < 2; i++) {
        const coordinator = await client.spawnAgent('analyst', `coordinator-${i}`, {
            role: 'swarm_coordinator',
            decision_making: 'consensus',
            max_subordinates: 5
        });
        swarm.coordinators.push(coordinator);
    }
    console.log(`  ✅ Spawned ${swarm.coordinators.length} coordinators`);
    
    // Spawn worker agents
    const workerTypes = ['coder', 'researcher', 'documenter'];
    for (let i = 0; i < 6; i++) {
        const type = workerTypes[i % workerTypes.length];
        const worker = await client.spawnAgent(type, `worker-${type}-${i}`, {
            swarm_role: 'worker',
            collaborative: true,
            skill_sharing: true
        });
        swarm.workers.push(worker);
    }
    console.log(`  ✅ Spawned ${swarm.workers.length} workers`);
    
    // Spawn reviewer agents
    for (let i = 0; i < 2; i++) {
        const reviewer = await client.spawnAgent('reviewer', `reviewer-${i}`, {
            standards: ['accuracy', 'completeness', 'clarity'],
            veto_power: true
        });
        swarm.reviewers.push(reviewer);
    }
    console.log(`  ✅ Spawned ${swarm.reviewers.length} reviewers`);
    
    // Store swarm structure
    await client.storeMemory('swarm_structure', swarm);
    
    // Create coordinated tasks
    const coordinatedTasks = [
        {
            id: 'task-cluster-1',
            description: 'Generate API documentation from source code',
            subtasks: [
                'Parse API endpoints',
                'Extract parameter schemas',
                'Generate OpenAPI spec',
                'Create interactive docs'
            ],
            coordinator: swarm.coordinators[0].agent_id,
            workers: swarm.workers.slice(0, 3).map(w => w.agent_id)
        },
        {
            id: 'task-cluster-2',
            description: 'Create comprehensive tutorials',
            subtasks: [
                'Identify common use cases',
                'Write step-by-step guides',
                'Create code examples',
                'Add interactive demos'
            ],
            coordinator: swarm.coordinators[1].agent_id,
            workers: swarm.workers.slice(3, 6).map(w => w.agent_id)
        }
    ];
    
    // Execute coordinated task clusters
    for (const cluster of coordinatedTasks) {
        console.log(`\n  📋 Executing: ${cluster.description}`);
        
        // Coordinator creates master task
        const masterTask = await client.createTask(
            'coordination',
            cluster.description,
            'critical',
            cluster.coordinator
        );
        
        // Workers create subtasks
        for (let i = 0; i < cluster.subtasks.length; i++) {
            const workerId = cluster.workers[i % cluster.workers.length];
            await client.createTask(
                'development',
                cluster.subtasks[i],
                'high',
                workerId
            );
        }
        
        console.log(`    ✅ Created ${cluster.subtasks.length + 1} coordinated tasks`);
    }
    
    // Monitor swarm behavior
    console.log('\n  👁️  Monitoring Swarm Behavior');
    
    client.on('ruv-swarm/event', (event) => {
        if (event.event.type === 'swarm_consensus') {
            console.log(`    🤝 Consensus reached: ${event.event.data.decision}`);
        } else if (event.event.type === 'task_handoff') {
            console.log(`    🔄 Task handoff: ${event.event.data.from} → ${event.event.data.to}`);
        }
    });
    
    await client.monitor(['swarm_consensus', 'task_handoff', 'conflict_resolution'], 10);
    
    // Optimize swarm performance
    const swarmOptimization = await client.optimize('throughput', {
        load_balancing: 'dynamic',
        communication_overhead: 'minimize',
        redundancy_level: 'adaptive'
    }, true);
    
    console.log('\n  ✅ Swarm optimization completed');
    console.log(`    Recommendations applied: ${swarmOptimization.recommendations.length}`);
    
    // Final swarm status
    const swarmStatus = await client.query({ swarm: true }, true);
    console.log('\n  📊 Final Swarm Status:');
    console.log(`    Active Agents: ${swarmStatus.total_agents}`);
    console.log(`    Tasks Completed: ${swarmStatus.completed_tasks}`);
    console.log(`    Coordination Efficiency: ${((swarmStatus.completed_tasks / swarmStatus.total_agents) * 100).toFixed(1)}%`);
}

// Main execution function
async function main() {
    console.log('🚀 RUV-SWARM MCP Workflow Examples');
    console.log('=' .repeat(50));
    
    const client = new RuvSwarmMCP();
    
    try {
        // Connect to MCP server
        await client.connect();
        
        // Run workflows based on command line argument
        const workflow = process.argv[2] || 'all';
        
        switch (workflow) {
            case '1':
            case 'webapp':
                await webAppDevelopmentWorkflow(client);
                break;
                
            case '2':
            case 'research':
                await aiResearchSwarmWorkflow(client);
                break;
                
            case '3':
            case 'cicd':
                await cicdPipelineWorkflow(client);
                break;
                
            case '4':
            case 'data':
                await dataAnalysisPipeline(client);
                break;
                
            case '5':
            case 'swarm':
                await swarmCoordinationWorkflow(client);
                break;
                
            case 'all':
            default:
                await webAppDevelopmentWorkflow(client);
                await aiResearchSwarmWorkflow(client);
                await cicdPipelineWorkflow(client);
                await dataAnalysisPipeline(client);
                await swarmCoordinationWorkflow(client);
                break;
        }
        
        // Display final statistics
        console.log('\n📊 Workflow Execution Summary');
        console.log('=' .repeat(50));
        
        const finalStatus = await client.query({}, true);
        console.log(`Total Agents Created: ${finalStatus.total_agents}`);
        console.log(`Total Tasks Executed: ${finalStatus.active_tasks + finalStatus.completed_tasks}`);
        console.log(`Tasks Completed: ${finalStatus.completed_tasks}`);
        
        // List all stored memory keys
        const agents = await client.listAgents(true);
        console.log(`\n📝 Stored Memory Keys: ${agents.count} agents tracked`);
        
    } catch (error) {
        console.error('❌ Error:', error);
    } finally {
        await client.disconnect();
    }
}

// Export for use in other modules
module.exports = {
    RuvSwarmMCP,
    webAppDevelopmentWorkflow,
    aiResearchSwarmWorkflow,
    cicdPipelineWorkflow,
    dataAnalysisPipeline,
    swarmCoordinationWorkflow
};

// Run if called directly
if (require.main === module) {
    console.log('\nUsage: node mcp-workflows.js [workflow]');
    console.log('Workflows:');
    console.log('  1 or webapp   - Web Application Development');
    console.log('  2 or research - AI Research Swarm');
    console.log('  3 or cicd     - CI/CD Pipeline');
    console.log('  4 or data     - Data Analysis Pipeline');
    console.log('  5 or swarm    - Swarm Coordination');
    console.log('  all           - Run all workflows (default)\n');
    
    main().catch(console.error);
}