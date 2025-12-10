const EDGE_FUNCTIONS = [
  {
    name: 'intent-detection',
    path: './intent-detection.ts',
    description: 'Detects intents in meeting transcripts using AI'
  },
  {
    name: 'meeting-info',
    path: './meeting-info.ts',
    description: 'Retrieves meeting information and summaries'
  },
  {
    name: 'webhook-handler',
    path: './webhook-handler.ts',
    description: 'Handles incoming webhooks for meeting events'
  }
];

interface DeploymentConfig {
  project?: string;
  region?: string;
  env?: Record<string, string>;
}

async function deployFunction(name: string, config: DeploymentConfig = {}) {
  const func = EDGE_FUNCTIONS.find(f => f.name === name);
  if (!func) {
    throw new Error(`Function ${name} not found`);
  }

  const deployCmd = new Deno.Command("deployctl", {
    args: [
      "deploy",
      "--project", config.project || "default",
      "--prod",
      func.path
    ],
    env: {
      ...config.env,
      DENO_DEPLOY_TOKEN: Deno.env.get("DENO_DEPLOY_TOKEN") || "",
    }
  });

  const { code, stdout, stderr } = await deployCmd.output();
  
  if (code === 0) {
    console.log(`Successfully deployed ${func.name}`);
    console.log(new TextDecoder().decode(stdout));
  } else {
    console.error(`Error deploying ${func.name}`);
    console.error(new TextDecoder().decode(stderr));
    throw new Error(`Deployment failed with code ${code}`);
  }
}

async function deployAll(config: DeploymentConfig = {}) {
  for (const func of EDGE_FUNCTIONS) {
    try {
      await deployFunction(func.name, config);
      console.log(`✓ ${func.name}: ${func.description}`);
    } catch (error) {
      console.error(`✗ ${func.name}: ${error.message}`);
    }
  }
}

if (import.meta.main) {
  const args = Deno.args;
  const config: DeploymentConfig = {
    project: args[1],
    region: args[2]
  };

  if (args[0] === "--all") {
    await deployAll(config);
  } else {
    await deployFunction(args[0], config);
  }
}
