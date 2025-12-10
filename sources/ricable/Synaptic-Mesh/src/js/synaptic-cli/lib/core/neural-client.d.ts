/**
 * Neural Client - Interface to neural network service
 */
export class NeuralClient {
    constructor(host?: string, port?: number);
    host: string;
    port: number;
    baseUrl: string;
    getModels(type?: null): Promise<never[]>;
    createModel(config: any): Promise<any>;
    trainModel(modelId: any, config: any): Promise<{
        id: string;
        modelId: any;
        config: any;
        status: string;
    }>;
    getTrainingStatus(trainingId: any): Promise<{
        status: string;
        currentEpoch: number;
        currentLoss: number;
        finalLoss: number;
        finalAccuracy: number;
        duration: number;
    }>;
    evaluateModel(modelId: any, config: any): Promise<{
        metrics: {
            accuracy: number;
            precision: number;
            recall: number;
            f1: number;
        };
        confusionMatrix: null;
    }>;
    predict(modelId: any, config: any): Promise<{
        predictions: never[];
        confidence: never[];
    }>;
    deleteModel(modelId: any): Promise<boolean>;
    exportModel(modelId: any, config: any): Promise<string>;
}
//# sourceMappingURL=neural-client.d.ts.map