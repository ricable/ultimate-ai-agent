import { HttpsProxyAgent } from "https-proxy-agent";
import OpenAI, { ClientOptions } from "openai";
import fs from "node:fs/promises";
import readline from "node:readline";
import {
  CONFIG_FILE,
  DEFAULT_CONFIG,
  HOME_DIR,
  PLUGINS_DIR,
} from "../constants";

export function getOpenAICommonOptions(): ClientOptions {
  const options: ClientOptions = {};
  if (process.env.PROXY_URL) {
    options.httpAgent = new HttpsProxyAgent(process.env.PROXY_URL);
  } else if (process.env.HTTPS_PROXY) {
    options.httpAgent = new HttpsProxyAgent(process.env.HTTPS_PROXY);
  } else if (process.env.https_proxy) {
    options.httpAgent = new HttpsProxyAgent(process.env.https_proxy);
  }
  return options;
}

const ensureDir = async (dir_path: string) => {
  try {
    await fs.access(dir_path);
  } catch {
    await fs.mkdir(dir_path, { recursive: true });
  }
};

export const initDir = async () => {
  await ensureDir(HOME_DIR);
  await ensureDir(PLUGINS_DIR);
};

const createReadline = () => {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
};

const question = (query: string): Promise<string> => {
  return new Promise((resolve) => {
    const rl = createReadline();
    rl.question(query, (answer) => {
      rl.close();
      resolve(answer);
    });
  });
};

const confirm = async (query: string): Promise<boolean> => {
  const answer = await question(query);
  return answer.toLowerCase() !== "n";
};

export const readConfigFile = async () => {
  try {
    const config = await fs.readFile(CONFIG_FILE, "utf-8");
    return JSON.parse(config);
  } catch {
    const apiKey = await question("Enter OPENAI_API_KEY: ");
    const baseUrl = await question("Enter OPENAI_BASE_URL: ");
    const model = await question("Enter OPENAI_MODEL: ");
    const config = Object.assign({}, DEFAULT_CONFIG, {
      OPENAI_API_KEY: apiKey,
      OPENAI_BASE_URL: baseUrl,
      OPENAI_MODEL: model,
    });
    await writeConfigFile(config);
    return config;
  }
};

export const writeConfigFile = async (config: any) => {
  await ensureDir(HOME_DIR);
  await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
};

export const initConfig = async () => {
  const config = await readConfigFile();
  Object.assign(process.env, config);
  return config;
};

export const createClient = (options: ClientOptions) => {
  const client = new OpenAI({
    ...options,
    ...getOpenAICommonOptions(),
  });
  return client;
};
