"""
Script to set up the database tables in Supabase using the Supabase MCP server.
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# SQL for creating the database tables and functions
SQL_SETUP = """
-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table rag_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on rag_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_rag_pages_metadata on rag_pages using gin (metadata);

-- Create an index on source for faster filtering
CREATE INDEX idx_rag_pages_source ON rag_pages ((metadata->>'source'));

-- Create a function to search for documentation chunks
create or replace function match_rag_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    1 - (rag_pages.embedding <=> query_embedding) as similarity
  from rag_pages
  where metadata @> filter
  order by rag_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table rag_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on rag_pages
  for select
  to public
  using (true);

-- Create a policy that allows anyone to insert
create policy "Allow public insert access"
  on rag_pages
  for insert
  to public
  with check (true);
"""

async def setup_database():
    """
    Set up the database tables and functions in Supabase.
    
    This function uses the Supabase MCP server to run the SQL setup script.
    """
    try:
        # In a real application, you would use the Supabase MCP server to run the SQL
        # For example:
        # result = await mcp2_apply_migration(name="rag_setup", query=SQL_SETUP)
        # print(f"Database setup completed: {result}")
        
        print("Database setup script generated.")
        print("To set up the database, use the Supabase MCP server to run the SQL script.")
        print("Example command:")
        print("mcp2_apply_migration(name=\"rag_setup\", query=SQL_SETUP)")
    except Exception as e:
        print(f"Error setting up database: {e}")


if __name__ == "__main__":
    asyncio.run(setup_database())
