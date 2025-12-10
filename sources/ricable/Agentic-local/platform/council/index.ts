/**
 * MLX Deep Council Module
 *
 * Distributed multi-model consensus system inspired by Andrej Karpathy's
 * LLM Council, optimized for Apple Silicon clusters using MLX.
 *
 * @module @edge-ai/mlx-deep-council
 */

// Main Council Class
export {
  MLXDeepCouncil,
  createLocalCouncil,
  createDistributedCouncil,
  createThunderboltCouncil,
} from './mlx-deep-council';

// Types
export type {
  CouncilMember,
  MemberMetrics,
  DistributedNode,
  CouncilConfig,
  CouncilQuery,
  IndividualResponse,
  PeerReview,
  ChairmanSynthesis,
  CouncilSession,
  SessionMetrics,
} from './mlx-deep-council';

// Re-export default
export { default } from './mlx-deep-council';
