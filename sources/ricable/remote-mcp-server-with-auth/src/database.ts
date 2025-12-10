import postgres from "postgres";

let dbInstance: postgres.Sql | null = null;

/**
 * Get database connection singleton
 * Following the pattern from BASIC-DB-MCP.md but adapted for PostgreSQL with connection pooling
 */
export function getDb(databaseUrl: string): postgres.Sql {
	if (!dbInstance) {
		dbInstance = postgres(databaseUrl, {
			// Connection pool settings for Cloudflare Workers
			max: 5, // Maximum 5 connections to fit within Workers' limit of 6 concurrent connections
			idle_timeout: 20,
			connect_timeout: 10,
			// Enable prepared statements for better performance
			prepare: true,
		});
	}
	return dbInstance;
}

/**
 * Close database connection pool
 * Call this when the Durable Object is shutting down
 */
export async function closeDb(): Promise<void> {
	if (dbInstance) {
		try {
			await dbInstance.end();
		} catch (error) {
			console.error('Error closing database connection:', error);
		} finally {
			dbInstance = null;
		}
	}
}

/**
 * Execute a database operation with proper connection management
 * Following the pattern from BASIC-DB-MCP.md but adapted for PostgreSQL
 */
export async function withDatabase<T>(
	databaseUrl: string,
	operation: (db: postgres.Sql) => Promise<T>
): Promise<T> {
	const db = getDb(databaseUrl);
	const startTime = Date.now();
	try {
		const result = await operation(db);
		const duration = Date.now() - startTime;
		console.log(`Database operation completed successfully in ${duration}ms`);
		return result;
	} catch (error) {
		const duration = Date.now() - startTime;
		console.error(`Database operation failed after ${duration}ms:`, error);
		// Re-throw the error so it can be caught by Sentry in the calling code
		throw error;
	}
	// Note: With PostgreSQL connection pooling, we don't close individual connections
	// They're returned to the pool automatically. The pool is closed when the Durable Object shuts down.
}

/**
 * SQL injection protection: Basic SQL keyword validation
 * This is a simple check - in production you should use parameterized queries
 */
export function validateSqlQuery(sql: string): { isValid: boolean; error?: string } {
	const trimmedSql = sql.trim().toLowerCase();
	
	// Check for empty queries
	if (!trimmedSql) {
		return { isValid: false, error: "SQL query cannot be empty" };
	}
	
	// Check for obviously dangerous patterns
	const dangerousPatterns = [
		/;\s*drop\s+/i,
		/;\s*delete\s+.*\s+where\s+1\s*=\s*1/i,
		/;\s*update\s+.*\s+set\s+.*\s+where\s+1\s*=\s*1/i,
		/;\s*truncate\s+/i,
		/;\s*alter\s+/i,
		/;\s*create\s+/i,
		/;\s*grant\s+/i,
		/;\s*revoke\s+/i,
		/xp_cmdshell/i,
		/sp_executesql/i,
	];
	
	for (const pattern of dangerousPatterns) {
		if (pattern.test(sql)) {
			return { isValid: false, error: "Query contains potentially dangerous SQL patterns" };
		}
	}
	
	return { isValid: true };
}

/**
 * Check if a SQL query is a write operation
 */
export function isWriteOperation(sql: string): boolean {
	const trimmedSql = sql.trim().toLowerCase();
	const writeKeywords = [
		'insert', 'update', 'delete', 'create', 'drop', 'alter', 
		'truncate', 'grant', 'revoke', 'commit', 'rollback'
	];
	
	return writeKeywords.some(keyword => trimmedSql.startsWith(keyword));
}

/**
 * Format database error for user-friendly display
 */
export function formatDatabaseError(error: unknown): string {
	if (error instanceof Error) {
		// Hide sensitive connection details
		if (error.message.includes('password')) {
			return "Database authentication failed. Please check your credentials.";
		}
		if (error.message.includes('timeout')) {
			return "Database connection timed out. Please try again.";
		}
		if (error.message.includes('connection')) {
			return "Unable to connect to database. Please check your connection string.";
		}
		return `Database error: ${error.message}`;
	}
	return "An unknown database error occurred.";
}