interface SupabaseConfig {
  projectId?: string;
  accessToken?: string;
}

interface DeploymentResult {
  success: boolean;
  message: string;
  functionUrl?: string;
  error?: string;
}

export class SupabaseDeployer {
  private config: SupabaseConfig;

  constructor(config: SupabaseConfig = {}) {
    this.config = {
      projectId: config.projectId || Deno.env.get("SUPABASE_PROJECT_ID"),
      accessToken: config.accessToken || Deno.env.get("SUPABASE_ACCESS_TOKEN")
    };
  }

  private async validateConfig(): Promise<boolean> {
    if (!this.config.projectId || !this.config.accessToken) {
      console.error('\x1b[31mError: Missing Supabase configuration\x1b[0m');
      console.error('Please set SUPABASE_PROJECT_ID and SUPABASE_ACCESS_TOKEN environment variables');
      return false;
    }
    return true;
  }

  private async executeCommand(command: string[]): Promise<{ success: boolean; output: string }> {
    try {
      const process = new Deno.Command("supabase", {
        args: command,
        env: {
          "SUPABASE_ACCESS_TOKEN": this.config.accessToken!
        }
      });

      const { code, stdout, stderr } = await process.output();
      const output = new TextDecoder().decode(code === 0 ? stdout : stderr);
      return { success: code === 0, output };
    } catch (error) {
      return { 
        success: false, 
        output: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async deployFunction(functionName: string): Promise<DeploymentResult> {
    if (!await this.validateConfig()) {
      return {
        success: false,
        message: 'Invalid configuration',
        error: 'Missing required environment variables'
      };
    }

    console.log(`\n\x1b[33mDeploying ${functionName} to Supabase...\x1b[0m`);

    // Verify function exists
    const functionPath = `src/packages/edge/${functionName}.ts`;
    try {
      await Deno.stat(functionPath);
    } catch {
      return {
        success: false,
        message: `Function ${functionName} not found`,
        error: `File ${functionPath} does not exist`
      };
    }

    // Create Supabase function directory
    const supabaseFunctionPath = `supabase/functions/${functionName}`;
    try {
      await Deno.mkdir(supabaseFunctionPath, { recursive: true });
    } catch (error) {
      return {
        success: false,
        message: 'Failed to create function directory',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }

    // Copy function file
    try {
      await Deno.copyFile(functionPath, `${supabaseFunctionPath}/index.ts`);
    } catch (error) {
      return {
        success: false,
        message: 'Failed to copy function file',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }

    // Deploy to Supabase
    const deployResult = await this.executeCommand([
      "functions", 
      "deploy", 
      functionName,
      "--project-ref", 
      this.config.projectId!
    ]);

    if (!deployResult.success) {
      return {
        success: false,
        message: 'Deployment failed',
        error: deployResult.output
      };
    }

    const functionUrl = `https://${this.config.projectId}.supabase.co/functions/v1/${functionName}`;
    return {
      success: true,
      message: `Successfully deployed ${functionName}`,
      functionUrl
    };
  }

  async listDeployedFunctions(): Promise<string[]> {
    if (!await this.validateConfig()) {
      return [];
    }

    const result = await this.executeCommand([
      "functions",
      "list",
      "--project-ref",
      this.config.projectId!
    ]);

    if (!result.success) {
      console.error('\x1b[31mError listing functions:\x1b[0m', result.output);
      return [];
    }

    // Parse function names from output
    const functions = result.output
      .split('\n')
      .slice(1) // Skip header
      .filter(line => line.trim())
      .map(line => line.split(/\s+/)[0]);

    return functions;
  }

  async getFunctionLogs(functionName: string): Promise<string[]> {
    if (!await this.validateConfig()) {
      return [];
    }

    const result = await this.executeCommand([
      "functions",
      "logs",
      functionName,
      "--project-ref",
      this.config.projectId!
    ]);

    if (!result.success) {
      console.error(`\x1b[31mError getting logs for ${functionName}:\x1b[0m`, result.output);
      return [];
    }

    return result.output.split('\n').filter(line => line.trim());
  }
}
