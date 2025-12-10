// File: frontend/src/components/agents/AgentCard.tsx
import { useState, useRef } from 'react';
import { useAGUI } from '@/hooks/useAGUI';
// Dummy components for UI, replace with Radix/Shadcn
const Card = ({ children, className }: any) => <div className={`border rounded-lg p-4 shadow-md bg-white ${className}`}>{children}</div>;
const CardHeader = ({ children }: any) => <div className="font-bold text-lg mb-2">{children}</div>;
const CardContent = ({ children }: any) => <div className="text-sm text-gray-700">{children}</div>;
const Input = (props: any) => <input className="border rounded w-full p-2 my-2" {...props} />;
const Button = (props: any) => <button className="bg-blue-500 text-white rounded px-4 py-2 w-full disabled:bg-gray-400" {...props} />;
const Badge = ({ children }: any) => <span className="bg-gray-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded-full">{children}</span>;


interface AgentCardProps {
  id: string;
  name: string;
  description: string;
  framework: string;
}

export function AgentCard({ id, name, description, framework }: AgentCardProps) {
  const { messages, isConnected, connectionStatus, sendMessage, reconnect, clearMessages } = useAGUI(id);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const chatWindowRef = useRef<HTMLDivElement>(null);

  const handleSend = async () => {
    if (input.trim() && !isSending) {
      setIsSending(true);
      try {
        await sendMessage(input, framework as 'auto' | 'copilot' | 'agno' | 'mastra');
        setInput('');
      } catch (error) {
        console.error('Failed to send message:', error);
      } finally {
        setIsSending(false);
      }
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus.state) {
      case 'connected': return 'bg-green-500';
      case 'connecting': 
      case 'reconnecting': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus.state) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'reconnecting': return `Reconnecting... (${connectionStatus.reconnectAttempts})`;
      case 'error': return `Error: ${connectionStatus.error || 'Unknown error'}`;
      default: return 'Disconnected';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <span>{name}</span>
          <Badge>{framework}</Badge>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <div className={`h-2 w-2 rounded-full ${getConnectionStatusColor()}`}></div>
          <span className="text-xs text-gray-600">{getConnectionStatusText()}</span>
          {connectionStatus.state === 'error' && (
            <Button
              onClick={reconnect}
              className="ml-auto px-2 py-1 text-xs bg-gray-500 hover:bg-gray-600"
            >
              Retry
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p className="mb-4">{description}</p>
        <div ref={chatWindowRef} className="h-48 overflow-y-scroll border rounded p-2 bg-gray-50 mb-2 scrollbar-visible">
          {messages.length === 0 ? (
            <div className="text-xs text-gray-400 p-2">No messages yet. Start a conversation!</div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className="text-xs p-1 border-b border-gray-200 last:border-b-0">
                <div className="flex justify-between items-start">
                  <strong className="text-blue-600">{msg.type}:</strong>
                  {msg.timestamp && (
                    <span className="text-gray-400 text-xs">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                {msg.content && <div className="mt-1">{msg.content}</div>}
                {msg.metadata && Object.keys(msg.metadata).length > 0 && (
                  <details className="mt-1">
                    <summary className="text-gray-500 cursor-pointer">Metadata</summary>
                    <pre className="text-xs text-gray-600 mt-1">
                      {JSON.stringify(msg.metadata, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))
          )}
        </div>
        <div className="flex space-x-2 mb-2">
          <Button
            onClick={clearMessages}
            className="px-2 py-1 text-xs bg-gray-500 hover:bg-gray-600"
            disabled={messages.length === 0}
          >
            Clear
          </Button>
        </div>
        <div className="flex space-x-2">
          <Input
            type="text"
            value={input}
            onChange={(e: any) => setInput(e.target.value)}
            onKeyDown={(e: any) => e.key === 'Enter' && handleSend()}
            placeholder="Chat with agent..."
            disabled={!isConnected || isSending}
          />
          <Button 
            onClick={handleSend} 
            disabled={!isConnected || isSending || !input.trim()}
          >
            {isSending ? 'Sending...' : 'Send'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}