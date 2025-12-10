
/**
 * Swarm Lifecycle Manager
 * Phase 4 Component: Ensures Production Autonomy and Self-Healing.
 * 
 * Responsibilities:
 * - Monitor Agent Health
 * - Auto-Respawn failed agents
 * - Maintain swarm density (scaling enforcement)
 */
export class SwarmManager {
    constructor(orchestrator) {
        this.orchestrator = orchestrator;
        this.monitoringInterval = null;
        this.targetDensity = {
            clusters: 0,
            sentinels: 0
        };
        this.isMonitoring = false;
    }

    /**
     * Start the autonomous monitoring loop
     * @param {number} intervalMs - Check interval
     */
    startMonitoring(intervalMs = 5000) {
        if (this.isMonitoring) return;
        this.isMonitoring = true;

        console.log('[SWARM_MANAGER] Starting autonomous monitoring...');

        // Ensure this doesn't block the process in a real deployment
        // For simulation/testing, we might want to manually trigger 'tick'
        this.monitoringInterval = setInterval(() => this.healthCheckTick(), intervalMs);
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        this.isMonitoring = false;
    }

    /**
     * Sets the target swarm configuration
     */
    setTargetDensity(clusters, sentinels) {
        this.targetDensity = { clusters, sentinels };
        console.log(`[SWARM_MANAGER] Target density updated: ${clusters} Clusters, ${sentinels} Sentinels.`);
    }

    /**
     * Single monitoring tick (public for testing)
     */
    async healthCheckTick() {
        const agents = this.orchestrator.activeAgents;
        const failedAgents = [];
        const currentCounts = { cluster_orchestrator: 0, sentinel: 0 };

        // 1. Check existing agents
        for (const [id, agent] of agents) {
            if (agent.type in currentCounts) {
                currentCounts[agent.type]++;
            }

            if (agent.status === 'failed' || agent.status === 'unresponsive') {
                console.warn(`[SWARM_MANAGER] Agent ${id} (${agent.type}) is unhealthy. Status: ${agent.status}`);
                failedAgents.push(id);
            }
        }

        // 2. Self-Healing: Respawn failed agents
        for (const agentId of failedAgents) {
            await this.healAgent(agentId);
        }

        // 3. Density Enforcement: Check if we are below targets
        // (Simplified logic: just check validation, spawning is handled by scaleSwarm usually)
        // But for autonomy, we should auto-correct if agents disappear
        if (currentCounts.cluster_orchestrator < this.targetDensity.clusters) {
            const missing = this.targetDensity.clusters - currentCounts.cluster_orchestrator;
            console.warn(`[SWARM_MANAGER] Cluster Orchestrator deficit detected. Spawning ${missing}...`);
            for (let i = 0; i < missing; i++) await this.orchestrator.spawnAgent('cluster_orchestrator', { reason: 'density_enforcement' });
        }

        if (currentCounts.sentinel < this.targetDensity.sentinels) {
            const missing = this.targetDensity.sentinels - currentCounts.sentinel;
            console.warn(`[SWARM_MANAGER] Sentinel deficit detected. Spawning ${missing}...`);
            for (let i = 0; i < missing; i++) await this.orchestrator.spawnAgent('sentinel', { reason: 'density_enforcement' });
        }
    }

    async healAgent(agentId) {
        const agent = this.orchestrator.activeAgents.get(agentId);
        if (!agent) return;

        console.log(`[SWARM_MANAGER] Initiating self-healing for ${agentId}...`);

        // 1. Remove dead agent
        this.orchestrator.activeAgents.delete(agentId);

        // 2. Respawn new agent with same config
        const newAgent = await this.orchestrator.spawnAgent(agent.type, agent.context);

        console.log(`[SWARM_MANAGER] Healed ${agent.type}. Old ID: ${agentId} -> New ID: ${newAgent.id}`);
    }
}
