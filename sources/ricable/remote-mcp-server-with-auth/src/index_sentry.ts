import * as Sentry from "@sentry/cloudflare";
import OAuthProvider from "@cloudflare/workers-oauth-provider";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { z } from "zod";
import { GitHubHandler } from "./github-handler";
import { validateSqlQuery, isWriteOperation, closeDb, withDatabase } from "./database";

// Context from the auth process, encrypted & stored in the auth token
// and provided to the DurableMCP as this.props
type Props = {
	login: string;
	name: string;
	email: string;
	accessToken: string;
};

const ALLOWED_USERNAMES = new Set<string>([
	// Add GitHub usernames of users who should have access to database write operations
	// For example: 'yourusername', 'coworkerusername'
	'coleam00'
]);

// Sentry configuration helper
function getSentryConfig(env: Env) {
	return {
		// You can disable Sentry by setting SENTRY_DSN to a falsey-value
		dsn: (env as any).SENTRY_DSN,
		// A sample rate of 1.0 means "capture all traces"
		tracesSampleRate: 1,
	};
}

// Error handling helper for MCP tools
function handleError(error: unknown): { content: Array<{ type: "text"; text: string; isError?: boolean }> } {
	const eventId = Sentry.captureException(error);

	const errorMessage = [
		"**Error**",
		"There was a problem with your request.",
		"Please report the following to the support team:",
		`**Event ID**: ${eventId}`,
		process.env.NODE_ENV !== "production"
			? error instanceof Error
				? error.message
				: String(error)
			: "",
	].join("\n\n");

	return {
		content: [
			{
				type: "text",
				text: errorMessage,
				isError: true,
			},
		],
	};
}

/**
 * Take the arguments from an MCP tool call and format
 * them in an OTel-safe way for tracing attributes.
 */
function extractMcpParameters(args: Record<string, any>) {
	return Object.fromEntries(
		Object.entries(args).map(([key, value]) => {
			return [`mcp.param.${key}`, JSON.stringify(value)];
		}),
	);
}

// Base MCP class without Sentry instrumentation
export class MyMCPBase extends McpAgent<Env, Record<string, never>, Props> {
	server = new McpServer({
		name: "PostgreSQL Database MCP Server with Sentry",
		version: "1.0.0",
	});

	constructor(state: DurableObjectState, env: Env) {
		super(state, env);
	}

	/**
	 * Cleanup database connections when Durable Object is shutting down
	 */
	async cleanup(): Promise<void> {
		try {
			await closeDb();
			console.log('Database connections closed successfully');
		} catch (error) {
			console.error('Error during database cleanup:', error);
			Sentry.captureException(error);
		}
	}

	/**
	 * Durable Objects alarm handler - used for cleanup
	 */
	async alarm(): Promise<void> {
		await this.cleanup();
	}

	// Helper function to register tools with Sentry instrumentation
	private registerTool(
		name: string,
		description: string,
		schema: any,
		handler: (args: any) => Promise<any>
	) {
		this.server.tool(name, description, schema, async (args: any) => {
			return await Sentry.startNewTrace(async () => {
				return await Sentry.startSpan(
					{
						name: `mcp.tool/${name}`,
						attributes: extractMcpParameters(args),
					},
					async (span) => {
						Sentry.setUser({
							username: this.props.login,
							email: this.props.email,
						});

						try {
							const result = await handler(args);
							return result;
						} catch (error) {
							span.setStatus({ code: 2 }); // error
							return handleError(error);
						}
					},
				);
			});
		});
	}

	async init() {
		// Tool 1: List Tables - Available to all authenticated users
		this.registerTool(
			"listTables",
			"Get a list of all tables in the database along with their column information. Use this first to understand the database structure before querying.",
			{},
			async () => {
				return await withDatabase((this.env as any).DATABASE_URL, async (db) => {
					// Single query to get all table and column information (using your working query)
					const columns = await db`
						SELECT 
							table_name, 
							column_name, 
							data_type, 
							is_nullable,
							column_default
						FROM information_schema.columns 
						WHERE table_schema = 'public' 
						ORDER BY table_name, ordinal_position
					`;
					
					// Group columns by table
					const tableMap = new Map();
					for (const col of columns) {
						// Use snake_case property names as returned by the SQL query
						if (!tableMap.has(col.table_name)) {
							tableMap.set(col.table_name, {
								name: col.table_name,
								schema: 'public',
								columns: []
							});
						}
						tableMap.get(col.table_name).columns.push({
							name: col.column_name,
							type: col.data_type,
							nullable: col.is_nullable === 'YES',
							default: col.column_default
						});
					}
					
					const tableInfo = Array.from(tableMap.values());
					
					return {
						content: [
							{
								type: "text",
								text: `**Database Tables and Schema**\n\n${JSON.stringify(tableInfo, null, 2)}\n\n**Total tables found:** ${tableInfo.length}\n\n**Note:** Use the \`queryDatabase\` tool to run SELECT queries, or \`executeDatabase\` tool for write operations (if you have write access).`
							}
						]
					};
				});
			}
		);

		// Tool 2: Query Database - Available to all authenticated users (read-only)
		this.registerTool(
			"queryDatabase",
			"Execute a read-only SQL query against the PostgreSQL database. This tool only allows SELECT statements and other read operations. All authenticated users can use this tool.",
			{
				sql: z.string().describe("The SQL query to execute. Only SELECT statements and read operations are allowed.")
			},
			async ({ sql }) => {
				// Validate the SQL query
				const validation = validateSqlQuery(sql);
				if (!validation.isValid) {
					throw new Error(`Invalid SQL query: ${validation.error}`);
				}
				
				// Check if it's a write operation
				if (isWriteOperation(sql)) {
					throw new Error("Write operations are not allowed with this tool. Use the executeDatabase tool if you have write permissions.");
				}
				
				return await withDatabase((this.env as any).DATABASE_URL, async (db) => {
					const results = await db.unsafe(sql);
					
					return {
						content: [
							{
								type: "text",
								text: `**Query Results**\n\`\`\`sql\n${sql}\n\`\`\`\n\n**Results:**\n\`\`\`json\n${JSON.stringify(results, null, 2)}\n\`\`\`\n\n**Rows returned:** ${Array.isArray(results) ? results.length : 1}`
							}
						]
					};
				});
			}
		);

		// Tool 3: Execute Database - Only available to privileged users (write operations)
		if (ALLOWED_USERNAMES.has(this.props.login)) {
			this.registerTool(
				"executeDatabase",
				"Execute any SQL statement against the PostgreSQL database, including INSERT, UPDATE, DELETE, and DDL operations. This tool is restricted to specific GitHub users and can perform write transactions. **USE WITH CAUTION** - this can modify or delete data.",
				{
					sql: z.string().describe("The SQL statement to execute. Can be any valid SQL including INSERT, UPDATE, DELETE, CREATE, etc.")
				},
				async ({ sql }) => {
					// Validate the SQL query
					const validation = validateSqlQuery(sql);
					if (!validation.isValid) {
						throw new Error(`Invalid SQL statement: ${validation.error}`);
					}
					
					const isWrite = isWriteOperation(sql);
					
					return await withDatabase((this.env as any).DATABASE_URL, async (db) => {
						const results = await db.unsafe(sql);
						
						const operationType = isWrite ? "Write Operation" : "Read Operation";
						
						return {
							content: [
								{
									type: "text",
									text: `**${operationType} Executed Successfully**\n\`\`\`sql\n${sql}\n\`\`\`\n\n**Results:**\n\`\`\`json\n${JSON.stringify(results, null, 2)}\n\`\`\`\n\n${isWrite ? '**⚠️ Database was modified**' : `**Rows returned:** ${Array.isArray(results) ? results.length : 1}`}\n\n**Executed by:** ${this.props.login} (${this.props.name})`
								}
							]
						};
					});
				}
			);
		}

		// Tool 4: Divide (for testing error handling)
		this.registerTool(
			"divide",
			"Divide two numbers to test error handling",
			{
				a: z.number().describe("The dividend (number to be divided)"),
				b: z.number().describe("The divisor (number to divide by)")
			},
			async ({ a, b }) => {
				// Intentionally cause an error when dividing by zero
				if (b === 0) {
					throw new Error("Cannot divide by zero");
				}
				
				const result = a / b;
				
				return {
					content: [
						{
							type: "text",
							text: `**Division Result**\n\n${a} ÷ ${b} = ${result}`
						}
					]
				};
			}
		);
	}
}

// Export the instrumented Durable Object
export const MyMCP = Sentry.instrumentDurableObjectWithSentry(
	getSentryConfig,
	MyMCPBase,
);

// Create the OAuth provider
const worker = new OAuthProvider({
	apiHandlers: {
		'/sse': MyMCP.serveSSE('/sse') as any,
		'/mcp': MyMCP.serve('/mcp') as any,
	},	
	authorizeEndpoint: "/authorize",
	clientRegistrationEndpoint: "/register",
	defaultHandler: GitHubHandler as any,
	tokenEndpoint: "/token",
});

// Export the worker wrapped with Sentry
export default Sentry.withSentry(
	getSentryConfig,
	worker,
) satisfies ExportedHandler<Env>;