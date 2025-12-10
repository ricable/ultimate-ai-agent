-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table rag_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on rag_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_rag_pages_metadata on rag_pages using gin (metadata);

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