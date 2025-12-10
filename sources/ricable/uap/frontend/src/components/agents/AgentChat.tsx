// AgentChat component with AG-UI protocol integration
import React, { useEffect, useState, useCallback } from 'react';
import { useAGUI } from '../../hooks/useAGUI';
import { AGUIEvent, TextMessageContentEvent, ToolCallStartEvent, ToolCallEndEvent } from '../../types/ag-ui';
import { X, MessageCircle, Settings, Activity, Zap, Send } from 'lucide-react';

interface AgentChatProps {
  agentId: string;
  agentName: string;
  framework: string;
  onClose: () => void;
}

export const AgentChat: React.FC<AgentChatProps> = ({ 
  agentId, 
  agentName, 
  framework, 
  onClose 
}) => {
  const { 
    messages, 
    connectionStatus, 
    isConnected, 
    sendMessage, 
    clearMessages 
  } = useAGUI(agentId);

  const [chatMessages, setChatMessages] = useState<any[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isSending, setIsSending] = useState(false);

  // Convert AG-UI events to CopilotKit message format
  const convertAGUIMessagesToCopilot = useCallback((aguiMessages: AGUIEvent[]) => {
    return aguiMessages
      .filter(msg => ['text_message_content', 'user_message', 'tool_call_start', 'tool_call_end'].includes(msg.type))
      .map((msg, index) => {
        switch (msg.type) {
          case 'user_message':
            return {
              id: `user-${index}`,
              role: 'user',
              content: msg.content || '',
              timestamp: msg.timestamp
            };
          case 'text_message_content':
            const textMsg = msg as TextMessageContentEvent;
            return {
              id: `assistant-${index}`,
              role: 'assistant',
              content: textMsg.content,
              timestamp: textMsg.timestamp,
              metadata: textMsg.metadata
            };
          case 'tool_call_start':
            const toolStart = msg as ToolCallStartEvent;
            return {
              id: `tool-start-${index}`,
              role: 'system',
              content: `🔧 Starting ${toolStart.metadata.toolName}...`,
              timestamp: toolStart.timestamp,
              metadata: toolStart.metadata
            };
          case 'tool_call_end':
            const toolEnd = msg as ToolCallEndEvent;
            return {
              id: `tool-end-${index}`,
              role: 'system',
              content: toolEnd.metadata.error 
                ? `❌ ${toolEnd.metadata.toolName} failed: ${toolEnd.metadata.error}`
                : `✅ ${toolEnd.metadata.toolName} completed`,
              timestamp: toolEnd.timestamp,
              metadata: toolEnd.metadata
            };
          default:
            return null;
        }
      })
      .filter(Boolean);
  }, []);

  // Update chat messages when AG-UI messages change
  useEffect(() => {
    const copilotMessages = convertAGUIMessagesToCopilot(messages);
    setChatMessages(copilotMessages);
  }, [messages, convertAGUIMessagesToCopilot]);

  // Handle typing indicator
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.type === 'agent_thinking') {
      setIsTyping(true);
    } else if (lastMessage?.type === 'text_message_content') {
      setIsTyping(false);
    }
  }, [messages]);

  // Handle message sending
  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || isSending) return;
    
    setIsSending(true);
    try {
      await sendMessage(inputMessage, framework as 'auto' | 'copilot' | 'agno' | 'mastra');
      setInputMessage('');
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsSending(false);
    }
  }, [inputMessage, isSending, sendMessage, framework]);

  const getStatusColor = () => {
    switch (connectionStatus.state) {
      case 'connected': return 'text-green-600';
      case 'connecting': 
      case 'reconnecting': return 'text-yellow-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus.state) {
      case 'connected': return <Zap className="h-4 w-4" />;
      case 'connecting': 
      case 'reconnecting': return <Activity className="h-4 w-4 animate-spin" />;
      case 'error': return <X className="h-4 w-4" />;
      default: return <MessageCircle className="h-4 w-4" />;
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-full max-w-4xl h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className={`flex items-center ${getStatusColor()}`}>
                {getStatusIcon()}
                <span className="ml-2 text-sm font-medium capitalize">
                  {connectionStatus.state}
                </span>
              </div>
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">{agentName}</h2>
              <p className="text-sm text-gray-500">
                {framework} • Agent ID: {agentId}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={clearMessages}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
              title="Clear chat history"
            >
              <Settings className="h-4 w-4" />
            </button>
            <button
              onClick={onClose}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Connection Status Banner */}
        {!isConnected && (
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <Activity className="h-5 w-5 text-yellow-400" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700">
                  {connectionStatus.state === 'connecting' && 'Connecting to agent...'}
                  {connectionStatus.state === 'reconnecting' && `Reconnecting... (attempt ${connectionStatus.reconnectAttempts})`}
                  {connectionStatus.state === 'error' && `Connection error: ${connectionStatus.error}`}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Chat Interface */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {isConnected ? (
            <div className="flex-1 flex flex-col">
              {/* Chat Messages */}
              <div className="flex-1 overflow-y-scroll p-4 space-y-4 scrollbar-visible">
                {chatMessages.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      Start a conversation
                    </h3>
                    <p className="text-gray-500">
                      Send a message to begin chatting with {agentName}
                    </p>
                  </div>
                ) : (
                  chatMessages.map((message, index) => (
                    <div
                      key={message.id || index}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                          message.role === 'user'
                            ? 'bg-blue-600 text-white'
                            : message.role === 'system'
                            ? 'bg-gray-200 text-gray-800 text-sm'
                            : 'bg-gray-100 text-gray-900'
                        }`}
                      >
                        <p className="text-sm">{message.content}</p>
                        {message.timestamp && (
                          <p className="text-xs mt-1 opacity-70">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 text-gray-900 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Message Input */}
              <div className="border-t border-gray-200 p-4">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                    placeholder={`Message ${agentName}...`}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={!isConnected || isSending}
                  />
                  <button
                    onClick={() => handleSendMessage()}
                    disabled={!isConnected || isSending || !inputMessage.trim()}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center"
                  >
                    {isSending ? (
                      <Activity className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4 animate-spin" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Connecting to {agentName}
                </h3>
                <p className="text-gray-500">
                  Establishing connection via AG-UI protocol...
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Custom Message Display for AG-UI Events */}
        {messages.length > 0 && (
          <div className="border-t border-gray-200 p-4 max-h-48 overflow-y-scroll bg-gray-50 scrollbar-visible">
            <h4 className="text-sm font-medium text-gray-900 mb-2">AG-UI Events:</h4>
            <div className="space-y-1">
              {messages.slice(-5).map((msg, index) => (
                <div key={index} className="text-xs">
                  <span className="font-medium text-blue-600">{msg.type}:</span>
                  {msg.content && (
                    <span className="ml-2 text-gray-700">{msg.content}</span>
                  )}
                  {msg.timestamp && (
                    <span className="ml-2 text-gray-400">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};