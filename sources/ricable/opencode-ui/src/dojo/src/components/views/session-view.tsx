"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { MessageSquare, ArrowLeft, Send, Bot, User, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useSessionStore, useActiveSession, useActiveSessionMessages } from "@/lib/session-store";
import { OpenCodeView } from "@/types/opencode";
import { safeToFixed } from "@/lib/utils";

interface SessionViewProps {
  sessionId: string;
  onViewChange: (view: OpenCodeView) => void;
}

export function SessionView({ sessionId, onViewChange }: SessionViewProps) {
  const session = useActiveSession();
  const messages = useActiveSessionMessages();
  const { isStreaming, actions } = useSessionStore();
  
  // Local state for message input
  const [messageInput, setMessageInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  useEffect(() => {
    // Load session messages
    actions.loadSessionMessages(sessionId);
  }, [sessionId, actions]);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const handleSendMessage = async () => {
    if (!messageInput.trim() || isSending || !session) {
      return;
    }

    const content = messageInput.trim();
    setIsSending(true);
    setSendError(null);

    try {
      // Clear input immediately for better UX
      setMessageInput("");
      
      // Send message through the session store
      await actions.sendMessage(sessionId, content);
    } catch (error) {
      console.error("Failed to send message:", error);
      setSendError(error instanceof Error ? error.message : "Failed to send message");
      // Restore message content on error
      setMessageInput(content);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const canSendMessage = messageInput.trim().length > 0 && !isSending && !isStreaming && session;

  return (
    <div className="flex-1 overflow-hidden bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => onViewChange("projects")}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">
                {session ? session.name : "OpenCode Session"}
              </h1>
              <p className="text-sm text-muted-foreground">
                {session ? `${session.provider}/${session.model}` : "Loading..."}
              </p>
            </div>
          </div>
          {session && (
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>{messages.length} messages</span>
              <span>${safeToFixed(session.total_cost, 4)} spent</span>
            </div>
          )}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center py-12"
            >
              <MessageSquare className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
              <p className="text-muted-foreground">
                Send a message to begin coding with OpenCode
              </p>
            </motion.div>
          ) : (
            messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex space-x-3 max-w-3xl ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  <div className="flex-shrink-0">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      message.role === 'user' 
                        ? 'bg-blue-500' 
                        : 'bg-gradient-to-br from-orange-400 to-red-500'
                    }`}>
                      {message.role === 'user' ? (
                        <User className="h-4 w-4 text-white" />
                      ) : (
                        <Bot className="h-4 w-4 text-white" />
                      )}
                    </div>
                  </div>
                  <Card className={`${message.role === 'user' ? 'bg-blue-50 dark:bg-blue-950/20' : ''}`}>
                    <CardContent className="p-4">
                      <div className="prose dark:prose-invert max-w-none">
                        <pre className="whitespace-pre-wrap font-sans text-sm">
                          {message.content}
                        </pre>
                      </div>
                      <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/50">
                        <span className="text-xs text-muted-foreground">
                          {formatTimestamp(message.timestamp)}
                        </span>
                        {message.cost != null && (
                          <span className="text-xs text-muted-foreground">
                            ${safeToFixed(message.cost, 4)}
                          </span>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            ))
          )}
          
          {isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="flex space-x-3 max-w-3xl">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                </div>
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-sm text-muted-foreground">Thinking...</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Message Input */}
      <div className="border-t border-border bg-card p-4">
        <div className="max-w-4xl mx-auto">
          {/* Error Display */}
          {sendError && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-3 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center space-x-2 text-red-700 dark:text-red-400"
            >
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span className="text-sm">{sendError}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSendError(null)}
                className="ml-auto h-6 px-2 text-red-700 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/20"
              >
                ×
              </Button>
            </motion.div>
          )}
          
          <div className="flex space-x-3">
            <div className="flex-1">
              <Textarea
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  !session 
                    ? "Loading session..." 
                    : isStreaming 
                    ? "AI is responding..." 
                    : "Type your message... (Enter to send, Shift+Enter for new line)"
                }
                disabled={!session || isStreaming || isSending}
                className="min-h-[80px] resize-none"
                rows={3}
              />
            </div>
            <Button 
              onClick={handleSendMessage}
              disabled={!canSendMessage}
              size="sm"
              className="self-end"
            >
              {isSending ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          
          {/* Status Information */}
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <div className="flex items-center space-x-4">
              {session && (
                <>
                  <span>Provider: {session.provider}</span>
                  <span>Model: {session.model}</span>
                </>
              )}
            </div>
            <div className="flex items-center space-x-2">
              {messageInput.length > 0 && (
                <span>{messageInput.length} characters</span>
              )}
              {isStreaming && (
                <span className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span>AI is responding...</span>
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}