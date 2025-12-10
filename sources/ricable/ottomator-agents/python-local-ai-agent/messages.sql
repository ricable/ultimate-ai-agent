   CREATE EXTENSION IF NOT EXISTS pgcrypto;

   CREATE TABLE n8n_chat_histories (
       id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       session_id TEXT NOT NULL,
       message JSONB NOT NULL
   );

   CREATE INDEX idx_messages_session_id ON messages(session_id);
   CREATE INDEX idx_messages_created_at ON messages(created_at);