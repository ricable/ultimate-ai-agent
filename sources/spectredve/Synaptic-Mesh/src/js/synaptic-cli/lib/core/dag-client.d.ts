/**
 * DAG Client - Interface to DAG workflow service
 */
export class DAGClient {
    constructor(host?: string, port?: number);
    host: string;
    port: number;
    baseUrl: string;
    getWorkflows(status?: null): Promise<never[]>;
    createWorkflow(config: any): Promise<any>;
    runWorkflow(workflowId: any, config: any): Promise<{
        id: string;
        workflowId: any;
        status: string;
        output: {
            result: string;
        } | null;
    }>;
    getExecutionStatus(executionId: any): Promise<{
        status: string;
        progress: number;
        currentNode: null;
        duration: number;
        nodesExecuted: number;
        nodes: {};
    }>;
    visualizeWorkflow(workflowId: any, config: any): Promise<"\n┌─────────┐    ┌─────────┐    ┌─────────┐\n│  Start  │───▶│ Process │───▶│   End   │\n└─────────┘    └─────────┘    └─────────┘\n" | "digraph { start -> process -> end; }">;
    deleteWorkflow(workflowId: any): Promise<boolean>;
}
//# sourceMappingURL=dag-client.d.ts.map