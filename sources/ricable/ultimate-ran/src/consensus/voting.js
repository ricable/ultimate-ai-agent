import { EventEmitter } from 'events';

/**
 * Agent Consensus Mechanism
 * Handles voting and resolution for unexpected behaviors.
 */
export class ConsensusMechanism extends EventEmitter {
    constructor(orchestrator) {
        super();
        this.orchestrator = orchestrator; // Reference to the orchestrator to contact agents
    }

    /**
     * Initiate a vote on a proposal or anomaly.
     * @param {string} proposalId
     * @param {string} description
     * @param {Array<string>} voters - List of agent IDs eligible to vote
     */
    async initiateVote(proposalId, description, voters) {
        console.log(`[CONSENSUS] Initiating vote for ${proposalId}: "${description}"`);

        const votes = {
            approve: 0,
            reject: 0,
            abstain: 0
        };

        const details = [];

        // Simulate collecting votes from agents
        for (const agentId of voters) {
            // In a real system, this would be an async call to the agent
            const vote = await this.collectVote(agentId, description);
            votes[vote.decision]++;
            details.push({ agentId, ...vote });
        }

        const result = this.adjudicate(votes);

        this.emit('consensus_reached', { proposalId, result, votes: details });
        console.log(`[CONSENSUS] Vote result for ${proposalId}: ${result}`);

        return result;
    }

    async collectVote(agentId, description) {
        // Simulation: Random voting logic
        // Guardian tends to be more conservative (reject)
        // Architect tends to approve optimization

        const decision = Math.random() > 0.2 ? 'approve' : 'reject';
        return { decision, reasoning: ' Automated heuristic decision' };
    }

    adjudicate(votes) {
        if (votes.reject > 0 && votes.reject >= votes.approve) {
            return 'rejected';
        }
        if (votes.approve > votes.reject) {
            return 'approved';
        }
        return 'deadlock';
    }
}
