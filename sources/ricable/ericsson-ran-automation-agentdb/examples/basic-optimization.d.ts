/**
 * Basic RAN Optimization Example
 *
 * This example demonstrates the core functionality of the Ericsson RAN
 * Optimization SDK with a simple optimization scenario.
 */
/**
 * Basic RAN metrics for optimization
 */
declare const basicRANMetrics: {
    energy_efficiency: number;
    mobility_performance: number;
    coverage_quality: number;
    capacity_utilization: number;
    user_experience: number;
    time_of_day: string;
    traffic_load: string;
    weather_conditions: string;
    event_type: string;
};
/**
 * Main optimization function
 */
declare function runBasicOptimization(): Promise<void>;
/**
 * Demonstrate progressive skill discovery
 */
declare function demonstrateSkillDiscovery(): Promise<void>;
/**
 * Demonstrate memory coordination
 */
declare function demonstrateMemoryCoordination(): Promise<void>;
export { runBasicOptimization, demonstrateSkillDiscovery, demonstrateMemoryCoordination, basicRANMetrics };
//# sourceMappingURL=basic-optimization.d.ts.map