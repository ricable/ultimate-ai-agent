"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadConfig = loadConfig;
exports.saveConfig = saveConfig;
exports.getConfigPath = getConfigPath;
exports.configExists = configExists;
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const CONFIG_DIR = '.synaptic';
const CONFIG_FILE = 'config.json';
async function loadConfig(projectPath = process.cwd()) {
    try {
        const configPath = path_1.default.join(projectPath, CONFIG_DIR, CONFIG_FILE);
        if (!(await fs_extra_1.default.pathExists(configPath))) {
            return null;
        }
        const config = await fs_extra_1.default.readJson(configPath);
        return config;
    }
    catch (error) {
        throw new Error(`Failed to load configuration: ${error.message}`);
    }
}
async function saveConfig(config, projectPath = process.cwd()) {
    try {
        const configDir = path_1.default.join(projectPath, CONFIG_DIR);
        const configPath = path_1.default.join(configDir, CONFIG_FILE);
        await fs_extra_1.default.ensureDir(configDir);
        await fs_extra_1.default.writeJson(configPath, config, { spaces: 2 });
    }
    catch (error) {
        throw new Error(`Failed to save configuration: ${error.message}`);
    }
}
async function getConfigPath(projectPath = process.cwd()) {
    return path_1.default.join(projectPath, CONFIG_DIR);
}
async function configExists(projectPath = process.cwd()) {
    const configPath = path_1.default.join(projectPath, CONFIG_DIR, CONFIG_FILE);
    return fs_extra_1.default.pathExists(configPath);
}
//# sourceMappingURL=loader.js.map