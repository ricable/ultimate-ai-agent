import sqlite3
from typing import List, Optional, Dict
import re
from contextlib import contextmanager

# Whitelist of allowed tables and their columns
ALLOWED_TABLES: Dict[str, List[str]] = {
    'users': ['id', 'name', 'email'],
    'profiles': ['id', 'bio', 'avatar'],
    'settings': ['id', 'preferences']
}

# Database configuration with validation
DB_CONFIG = {
    'timeout': max(min(float(5.0), 30.0), 1.0),  # Clamp between 1-30 seconds
    'isolation_level': 'EXCLUSIVE',
    'check_same_thread': False
}

def validate_db_config(config: Dict) -> None:
    """Validate database configuration parameters"""
    if not isinstance(config.get('timeout'), (int, float)):
        raise DatabaseError("Invalid timeout value")
    if config.get('isolation_level') not in ('DEFERRED', 'IMMEDIATE', 'EXCLUSIVE'):
        raise DatabaseError("Invalid isolation level")
    if not isinstance(config.get('check_same_thread'), bool):
        raise DatabaseError("Invalid check_same_thread value")

# Cache for prepared statements
STMT_CACHE = {}

def get_prepared_statement(conn: sqlite3.Connection, sql: str, params: tuple = None) -> sqlite3.Cursor:
    """
    Get or create a prepared statement with proper parameter binding and caching
    
    Args:
        conn: Database connection
        sql: SQL query string
        params: Query parameters
        
    Returns:
        sqlite3.Cursor: Prepared statement cursor
    """
    if not isinstance(sql, str):
        raise DatabaseError("SQL query must be a string")
        
    cursor = conn.cursor()
    
    # Use query template as cache key
    cache_key = hash(sql)
    
    if cache_key not in STMT_CACHE:
        try:
            # Validate the SQL before caching
            cursor.execute("EXPLAIN QUERY PLAN " + sql, params or ())
            STMT_CACHE[cache_key] = sql
        except sqlite3.Error as e:
            raise DatabaseError(f"Invalid SQL statement: {str(e)}")
    
    try:
        # Execute with parameters using proper binding
        if params:
            if not isinstance(params, (tuple, list)):
                raise DatabaseError("Query parameters must be tuple or list")
            cursor.execute(STMT_CACHE[cache_key], params)
        else:
            cursor.execute(STMT_CACHE[cache_key])
    except sqlite3.Error as e:
        raise DatabaseError(f"Error executing statement: {str(e)}")
        
    return cursor

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

@contextmanager
def get_db_connection():
    """
    Context manager for database connections with proper configuration
    """
    conn = None
    try:
        conn = sqlite3.connect('users.db', **DB_CONFIG)
        yield conn
    finally:
        if conn:
            conn.close()

def validate_table_name(table_name: str) -> str:
    """
    Validate if table name is in the allowed list with strict validation
    
    Args:
        table_name: The table name to validate
        
    Returns:
        str: The validated table name
        
    Raises:
        DatabaseError: If table name is invalid
    """
    if not isinstance(table_name, str):
        raise DatabaseError("Table name must be a string")
        
    table_name = table_name.lower().strip()
    
    # Direct whitelist check
    if table_name not in ALLOWED_TABLES:
        raise DatabaseError("Table not in allowed list")
    
    # Enhanced format validation with strict pattern
    if not re.match(r'^[a-z][a-z0-9_]{1,62}[a-z0-9]$', table_name, re.ASCII):
        raise DatabaseError("Invalid table name format")
        
    # Additional security checks
    if any(char in table_name for char in "\"';-/\\"):
        raise DatabaseError("Invalid characters in table name")
        
    # Prevent SQL injection attempts
    if any(keyword.lower() in table_name.lower() 
           for keyword in ['select', 'insert', 'update', 'delete', 'drop', 'union']):
        raise DatabaseError("Invalid table name - contains SQL keywords")
    
    # Length check
    if len(table_name) > 63:
        raise DatabaseError("Table name too long")
        
    # Verify table exists in database with transaction
    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN TRANSACTION")
            cursor = conn.cursor()
            # Use prepared statement
            stmt = get_prepared_statement(
                conn,
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? AND sql NOT LIKE '%--'",
                (table_name,)
            )
            if not stmt.fetchone():
                raise DatabaseError(f"Table '{table_name}' does not exist in database")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        
    return table_name

def build_secure_column_query(columns: List[str], table_name: str) -> tuple[str, List[str]]:
    """
    Build a secure parameterized query for column selection
    
    Args:
        columns: List of column names
        table_name: Table name
        
    Returns:
        Tuple of (query_string, parameters)
    """
    validated_cols = validate_columns(table_name, columns)
    validated_table = validate_table_name(table_name)
    
    # Build column list with proper quoting
    column_list = ', '.join(f'"{col}"' for col in validated_cols)
    
    # Use a fixed query structure without string interpolation
    query = f"""
        SELECT {column_list} 
        FROM "{validated_table}"
        WHERE active = 1 
        AND deleted_at IS NULL
    """
    return query, []  # No parameters needed since we validated and quoted everything

def validate_columns(table_name: str, columns: List[str]) -> List[str]:
    """
    Validate if requested columns are allowed for the table and contain only valid characters
    
    Args:
        table_name: The table to validate columns against
        columns: List of column names to validate
        
    Returns:
        List[str]: List of validated column names
        
    Raises:
        DatabaseError: If any column is invalid
    """
    if not isinstance(columns, list):
        raise DatabaseError("Columns must be provided as a list")
        
    if not columns:
        raise DatabaseError("At least one column must be specified")
        
    if not all(isinstance(col, str) for col in columns):
        raise DatabaseError("All column names must be strings")
    
    # Get allowed columns for the table
    allowed_cols = ALLOWED_TABLES.get(table_name.lower(), [])
    if not allowed_cols:
        raise DatabaseError(f"No columns defined for table '{table_name}'")
    
    validated_columns = []
    for col in columns:
        col = col.lower().strip()
        
        # Whitelist check
        if col not in allowed_cols:
            raise DatabaseError(f"Column '{col}' not allowed for table '{table_name}'")
            
        # Enhanced format validation
        if not re.match(r'^[a-z][a-z0-9_]{0,63}$', col):
            raise DatabaseError(f"Invalid column name format: {col}")
            
        # Additional security checks
        if any(char in col for char in "\"';-/\\"):
            raise DatabaseError("Invalid characters in column name")
            
        # Prevent SQL injection via column names
        if any(keyword.lower() in col.lower() 
               for keyword in ['select', 'insert', 'update', 'delete', 'drop', 'union']):
            raise DatabaseError("Invalid column name - contains SQL keywords")
            
        validated_columns.append(col)
    
    return validated_columns

def validate_user_id(user_id: str) -> bool:
    """
    Validate if user_id contains only digits and is within reasonable length
    """
    try:
        # Convert to string if not already
        user_id_str = str(user_id)
        # Check format and length
        if not re.match(r'^\d{1,10}$', user_id_str):
            return False
        # Verify value range
        user_id_int = int(user_id_str)
        return 0 < user_id_int < 1000000000
    except (ValueError, TypeError):
        return False

def get_user_data(user_id: str, table_name: str, columns: Optional[List[str]] = None) -> Optional[List[tuple]]:
    """
    Safely get user data using fully parameterized queries and prepared statements
    
    Args:
        user_id: The user ID to query
        table_name: The table to query from
        columns: Optional list of columns to retrieve
        
    Returns:
        List of tuples containing user data or None if error occurs
        
    Raises:
        DatabaseError: If invalid input or database error occurs
    """
    try:
        # Validate inputs
        if not all(isinstance(x, str) for x in [table_name, user_id]):
            raise DatabaseError("Invalid input types")
            
        table_name = validate_table_name(table_name)
        
        if not validate_user_id(user_id):
            raise DatabaseError("Invalid user ID format")
        
        # Use all columns if none specified
        if columns is None:
            columns = ALLOWED_TABLES[table_name]
        cols = validate_columns(table_name, columns)
        
        with get_db_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                # Build query using proper parameterization
                placeholders = ','.join(['?'] * len(cols))
                query = """
                    SELECT * FROM (
                        SELECT ? as col_name
                        UNION ALL
                    ) cols 
                    INNER JOIN (
                        SELECT * FROM ? 
                        WHERE id = ?
                        AND active = 1 
                        AND deleted_at IS NULL
                    ) data
                """
                
                # Prepare parameters including column names and table
                params = cols + [table_name, user_id]
                
                # Get prepared statement with parameters
                stmt = get_prepared_statement(conn, query, params)
                results = stmt.fetchall()
                
                conn.commit()
                return results if results else None
                
            except Exception:
                conn.rollback()
                raise
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error occurred: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def search_users(keyword: str, columns: Optional[List[str]] = None) -> Optional[List[tuple]]:
    """
    Safely search users using fully parameterized queries with strict input validation
    
    Args:
        keyword: Search term for user names
        columns: Optional list of columns to retrieve
        
    Returns:
        List of tuples containing matching users or None if error occurs
        
    Raises:
        DatabaseError: If database error occurs or invalid input
    """
    try:
        if not isinstance(keyword, str):
            raise DatabaseError("Search keyword must be a string")
            
        # Stricter keyword validation
        keyword = keyword.strip()
        if not keyword:
            raise DatabaseError("Search keyword cannot be empty")
            
        if len(keyword) < 3 or len(keyword) > 50:
            raise DatabaseError("Search keyword must be between 3 and 50 characters")
            
        # Only allow letters, numbers, spaces and hyphens
        if not re.match(r'^[a-zA-Z0-9\s-]+$', keyword):
            raise DatabaseError("Search keyword can only contain letters, numbers, spaces and hyphens")
            
        # Prevent common SQL injection patterns
        if any(pattern in keyword.lower() for pattern in ['union', 'select', '--', ';', '/*', '*/']):
            raise DatabaseError("Invalid search keyword")
        
        # Use all columns if none specified
        if columns is None:
            columns = ALLOWED_TABLES['users']
        cols = validate_columns('users', columns)
        
        with get_db_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                # Build the column list safely
                column_list = ', '.join(f'"{col}"' for col in cols)
                
                # Use a safer parameterized query approach
                query = """
                    WITH RECURSIVE split(word, str) AS (
                        SELECT '', ? || ' '
                        UNION ALL
                        SELECT substr(str, 0, instr(str, ' ')),
                        substr(str, instr(str, ' ')+1)
                        FROM split WHERE str!=''
                    )
                    SELECT DISTINCT u.*
                    FROM users u
                    WHERE EXISTS (
                        SELECT 1 FROM split 
                        WHERE word != ''
                        AND u.name LIKE '%' || replace(replace(replace(word, 
                            '\\', '\\\\'), 
                            '%', '\\%'), 
                            '_', '\\_') || '%' ESCAPE '\\'
                    )
                    AND u.active = 1
                    ORDER BY u.id ASC
                    LIMIT 100
                """
                
                # Pass the keyword directly as parameter
                params = (keyword,)
                
                # Get prepared statement with parameters
                stmt = get_prepared_statement(
                    conn,
                    query,
                    params=params
                )
                results = stmt.fetchall()
                
                conn.commit()
                return results if results else None
                
            except Exception:
                conn.rollback()
                raise
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Database error occurred: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    try:
        # Safe usage examples
        print(get_user_data("123", "users"))
        print(search_users("John"))
    except DatabaseError as e:
        print(f"Error: {e}")
