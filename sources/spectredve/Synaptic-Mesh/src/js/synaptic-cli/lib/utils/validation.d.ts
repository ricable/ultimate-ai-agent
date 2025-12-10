export function validateConfig(config: any): {
    valid: boolean;
    errors: string[];
};
export function validateEnvironment(): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
}>;
export function validateWorkflowDefinition(definition: any): {
    valid: boolean;
    errors: string[];
};
export function validateNeuralModelConfig(config: any): {
    valid: boolean;
    errors: string[];
};
//# sourceMappingURL=validation.d.ts.map