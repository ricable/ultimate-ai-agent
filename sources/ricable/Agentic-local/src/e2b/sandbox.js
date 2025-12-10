/**
 * =============================================================================
 * E2B Secure Sandbox Integration
 * Cloud-based secure code execution using E2B Firecracker microVMs
 * =============================================================================
 */

import EventEmitter from 'events';

/**
 * E2BSandbox - Secure code execution via E2B cloud
 */
export class E2BSandbox extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            apiKey: process.env.E2B_API_KEY,
            baseUrl: 'https://api.e2b.dev',
            defaultTemplate: 'python-3.10',
            timeout: 300000, // 5 minutes
            keepAlive: 60000, // 1 minute
            fallbackToLocal: true,
            ...config
        };

        this.activeSandboxes = new Map();
        this.templates = new Map([
            ['python', 'python-3.10'],
            ['python-data', 'python-data-science'],
            ['nodejs', 'nodejs-18'],
            ['bash', 'base']
        ]);
    }

    /**
     * Initialize E2B sandbox integration
     */
    async initialize() {
        // Verify API key
        if (!this.config.apiKey) {
            if (this.config.fallbackToLocal) {
                console.warn('E2B API key not set, falling back to local sandbox');
                this.emit('fallback-local');
                return true;
            }
            throw new Error('E2B_API_KEY environment variable is required');
        }

        // Verify API connectivity
        try {
            const response = await fetch(`${this.config.baseUrl}/health`, {
                headers: this.getHeaders()
            });

            if (!response.ok) {
                throw new Error(`E2B API health check failed: ${response.status}`);
            }

            this.emit('initialized');
            return true;
        } catch (error) {
            if (this.config.fallbackToLocal) {
                console.warn('E2B API not reachable, falling back to local sandbox');
                this.emit('fallback-local');
                return true;
            }
            throw error;
        }
    }

    /**
     * Get API headers
     */
    getHeaders() {
        return {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`
        };
    }

    /**
     * Create a new sandbox instance
     * @param {Object} options - Sandbox options
     * @returns {Promise<Object>} Sandbox instance
     */
    async createSandbox(options = {}) {
        const template = options.template || this.config.defaultTemplate;
        const templateId = this.templates.get(template) || template;

        const response = await fetch(`${this.config.baseUrl}/sandboxes`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                template: templateId,
                metadata: options.metadata || {},
                timeout: options.timeout || this.config.timeout
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(`Failed to create sandbox: ${error.message || response.status}`);
        }

        const sandbox = await response.json();

        this.activeSandboxes.set(sandbox.id, {
            ...sandbox,
            createdAt: new Date(),
            template: templateId
        });

        this.emit('sandbox-created', { id: sandbox.id, template: templateId });
        return sandbox;
    }

    /**
     * Execute code in a sandbox
     * @param {string} code - Code to execute
     * @param {Object} options - Execution options
     * @returns {Promise<Object>} Execution result
     */
    async executeCode(code, options = {}) {
        const {
            language = 'python',
            sandboxId = null,
            timeout = 60000,
            env = {},
            files = []
        } = options;

        let sandbox;
        let shouldDestroy = false;

        // Use existing sandbox or create new one
        if (sandboxId && this.activeSandboxes.has(sandboxId)) {
            sandbox = this.activeSandboxes.get(sandboxId);
        } else {
            sandbox = await this.createSandbox({ template: language });
            shouldDestroy = !options.keepAlive;
        }

        this.emit('execution-started', { sandboxId: sandbox.id, language });

        try {
            // Upload any required files
            for (const file of files) {
                await this.uploadFile(sandbox.id, file.path, file.content);
            }

            // Execute the code
            const result = await this.runCode(sandbox.id, code, { language, timeout, env });

            this.emit('execution-completed', {
                sandboxId: sandbox.id,
                success: !result.error,
                duration: result.duration
            });

            return result;
        } finally {
            // Cleanup if not keeping alive
            if (shouldDestroy) {
                await this.destroySandbox(sandbox.id);
            }
        }
    }

    /**
     * Run code in sandbox
     */
    async runCode(sandboxId, code, options) {
        const startTime = Date.now();

        const response = await fetch(`${this.config.baseUrl}/sandboxes/${sandboxId}/code/run`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                code,
                language: options.language,
                timeout: options.timeout,
                env: options.env
            })
        });

        const duration = Date.now() - startTime;

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            return {
                success: false,
                error: error.message || `Execution failed: ${response.status}`,
                stdout: '',
                stderr: error.stderr || '',
                duration
            };
        }

        const result = await response.json();

        return {
            success: !result.error,
            stdout: result.stdout || '',
            stderr: result.stderr || '',
            error: result.error || null,
            exitCode: result.exitCode,
            artifacts: result.artifacts || [],
            duration
        };
    }

    /**
     * Upload file to sandbox
     */
    async uploadFile(sandboxId, path, content) {
        const response = await fetch(`${this.config.baseUrl}/sandboxes/${sandboxId}/files`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({ path, content })
        });

        if (!response.ok) {
            throw new Error(`Failed to upload file: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Download file from sandbox
     */
    async downloadFile(sandboxId, path) {
        const response = await fetch(
            `${this.config.baseUrl}/sandboxes/${sandboxId}/files?path=${encodeURIComponent(path)}`,
            { headers: this.getHeaders() }
        );

        if (!response.ok) {
            throw new Error(`Failed to download file: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Execute shell command in sandbox
     */
    async executeCommand(sandboxId, command, options = {}) {
        const response = await fetch(`${this.config.baseUrl}/sandboxes/${sandboxId}/commands`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                command,
                timeout: options.timeout || 60000,
                env: options.env || {}
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(`Command execution failed: ${error.message || response.status}`);
        }

        return response.json();
    }

    /**
     * Install packages in sandbox
     */
    async installPackages(sandboxId, packages, options = {}) {
        const { packageManager = 'pip' } = options;

        let command;
        switch (packageManager) {
            case 'pip':
                command = `pip install ${packages.join(' ')}`;
                break;
            case 'npm':
                command = `npm install ${packages.join(' ')}`;
                break;
            case 'apt':
                command = `apt-get update && apt-get install -y ${packages.join(' ')}`;
                break;
            default:
                throw new Error(`Unknown package manager: ${packageManager}`);
        }

        return this.executeCommand(sandboxId, command, { timeout: 300000 });
    }

    /**
     * Destroy sandbox
     */
    async destroySandbox(sandboxId) {
        try {
            await fetch(`${this.config.baseUrl}/sandboxes/${sandboxId}`, {
                method: 'DELETE',
                headers: this.getHeaders()
            });

            this.activeSandboxes.delete(sandboxId);
            this.emit('sandbox-destroyed', { id: sandboxId });
        } catch (error) {
            console.warn(`Failed to destroy sandbox ${sandboxId}:`, error.message);
        }
    }

    /**
     * Get sandbox status
     */
    async getSandboxStatus(sandboxId) {
        const response = await fetch(`${this.config.baseUrl}/sandboxes/${sandboxId}`, {
            headers: this.getHeaders()
        });

        if (!response.ok) {
            throw new Error(`Failed to get sandbox status: ${response.status}`);
        }

        return response.json();
    }

    /**
     * List active sandboxes
     */
    getActiveSandboxes() {
        return Array.from(this.activeSandboxes.values());
    }

    /**
     * Cleanup all sandboxes
     */
    async cleanup() {
        const promises = [];

        for (const sandboxId of this.activeSandboxes.keys()) {
            promises.push(this.destroySandbox(sandboxId));
        }

        await Promise.all(promises);
        this.emit('cleanup-completed');
    }

    /**
     * Shutdown
     */
    async shutdown() {
        await this.cleanup();
        this.emit('shutdown');
    }
}

/**
 * HybridSandbox - Combines E2B cloud with local Docker fallback
 */
export class HybridSandbox extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            preferLocal: config.preferLocal || false,
            localForSimple: config.localForSimple || true,
            cloudForDataScience: config.cloudForDataScience || true,
            ...config
        };

        this.e2b = new E2BSandbox(config.e2b || {});
        this.localSandbox = null;
    }

    /**
     * Initialize hybrid sandbox
     */
    async initialize() {
        // Initialize E2B
        await this.e2b.initialize().catch(err => {
            console.warn('E2B initialization failed:', err.message);
        });

        // Initialize local sandbox if available
        try {
            const { DockerSandbox } = await import('../sandbox/docker-sandbox.js');
            this.localSandbox = new DockerSandbox(this.config.local || {});
            await this.localSandbox.initialize();
        } catch (error) {
            console.warn('Local sandbox not available:', error.message);
        }

        this.emit('initialized');
    }

    /**
     * Execute code with intelligent routing
     */
    async executeCode(code, options = {}) {
        const analysis = this.analyzeCode(code, options.language || 'python');

        // Determine best execution environment
        let useCloud = false;

        if (this.config.cloudForDataScience && analysis.requiresDataScience) {
            useCloud = true;
        }

        if (analysis.requiresNetwork && !this.localSandbox?.config?.networkEnabled) {
            useCloud = true;
        }

        if (this.config.preferLocal && this.localSandbox) {
            useCloud = false;
        }

        // Execute in chosen environment
        if (useCloud && this.e2b.config.apiKey) {
            this.emit('routing', { destination: 'cloud', reason: 'capability-match' });
            return this.e2b.executeCode(code, options);
        } else if (this.localSandbox) {
            this.emit('routing', { destination: 'local', reason: 'cost-optimization' });
            return this.localSandbox.execute(code, options);
        } else {
            throw new Error('No sandbox available');
        }
    }

    /**
     * Analyze code to determine requirements
     */
    analyzeCode(code, language) {
        const analysis = {
            requiresDataScience: false,
            requiresNetwork: false,
            estimatedComplexity: 'simple',
            packages: []
        };

        if (language === 'python') {
            // Check for data science imports
            const dataScienceLibs = ['pandas', 'numpy', 'scipy', 'sklearn', 'tensorflow', 'torch', 'matplotlib'];
            for (const lib of dataScienceLibs) {
                if (code.includes(`import ${lib}`) || code.includes(`from ${lib}`)) {
                    analysis.requiresDataScience = true;
                    analysis.packages.push(lib);
                }
            }

            // Check for network operations
            if (code.includes('requests') || code.includes('urllib') || code.includes('http')) {
                analysis.requiresNetwork = true;
            }

            // Estimate complexity
            if (analysis.requiresDataScience || code.length > 5000) {
                analysis.estimatedComplexity = 'complex';
            } else if (code.length > 1000) {
                analysis.estimatedComplexity = 'medium';
            }
        }

        return analysis;
    }

    /**
     * Shutdown
     */
    async shutdown() {
        await this.e2b.shutdown();
        if (this.localSandbox) {
            await this.localSandbox.shutdown();
        }
        this.emit('shutdown');
    }
}

export default E2BSandbox;
