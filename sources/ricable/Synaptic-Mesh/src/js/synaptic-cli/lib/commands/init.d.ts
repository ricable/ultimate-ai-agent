import { Command } from 'commander';
export interface InitOptions {
    name?: string;
    port?: number;
    network?: string;
    quantumResistant?: boolean;
    interactive?: boolean;
    force?: boolean;
}
export declare function initCommand(): Command;
//# sourceMappingURL=init.d.ts.map