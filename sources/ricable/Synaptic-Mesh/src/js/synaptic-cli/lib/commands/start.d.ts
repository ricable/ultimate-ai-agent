import { Command } from 'commander';
export interface StartOptions {
    port?: number;
    daemon?: boolean;
    ui?: boolean;
    metrics?: boolean;
}
export declare function startCommand(): Command;
//# sourceMappingURL=start.d.ts.map