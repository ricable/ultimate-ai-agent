import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useRoute, useNavigation } from '@react-navigation/native';
import { useNetwork } from '../hooks/useNetwork';
import { SyncService } from '../services/SyncService';
import { APIClient } from '../services/APIClient';
import { Logger } from '../utils/Logger';

const { width, height } = Dimensions.get('window');

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  status: 'sent' | 'pending' | 'failed';
  agentType?: string;
}

interface ChatScreenProps {
  agentId: string;
  agentName: string;
}

const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user';
  const getStatusIcon = () => {
    switch (message.status) {
      case 'pending':
        return <ActivityIndicator size="small" color="#6B7280" />;
      case 'failed':
        return <Icon name="error" size={16} color="#EF4444" />;
      case 'sent':
        return <Icon name="done" size={16} color="#10B981" />;
      default:
        return null;
    }
  };

  return (
    <View style={[styles.messageContainer, isUser ? styles.userMessage : styles.agentMessage]}>
      <View style={[styles.messageBubble, isUser ? styles.userBubble : styles.agentBubble]}>
        <Text style={[styles.messageText, isUser ? styles.userText : styles.agentText]}>
          {message.content}
        </Text>
        <View style={styles.messageFooter}>
          <Text style={[styles.timestamp, isUser ? styles.userTimestamp : styles.agentTimestamp]}>
            {message.timestamp.toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </Text>
          {isUser && (
            <View style={styles.statusIcon}>
              {getStatusIcon()}
            </View>
          )}
        </View>
      </View>
    </View>
  );
};

const ChatScreen: React.FC = () => {
  const route = useRoute();
  const navigation = useNavigation();
  const { agentId, agentName } = route.params as ChatScreenProps;
  const { isOnline } = useNetwork();
  const scrollViewRef = useRef<ScrollView>(null);
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [agentTyping, setAgentTyping] = useState(false);

  useEffect(() => {
    // Load chat history
    loadChatHistory();
    
    // Set navigation title
    navigation.setOptions({
      headerTitle: agentName,
      headerShown: true,
      headerBackTitleVisible: false,
      headerRight: () => (
        <TouchableOpacity onPress={handleInfoPress}>
          <Icon name="info" size={24} color="#2563EB" />
        </TouchableOpacity>
      ),
    });
  }, [agentId, agentName]);

  const loadChatHistory = async () => {
    try {
      // In a real app, this would load from local storage or API
      const history = await APIClient.getChatHistory(agentId);
      setMessages(history || []);
    } catch (error) {
      Logger.error('ChatScreen', 'Failed to load chat history', error);
      // Load from local storage as fallback
      // Implementation would go here
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const userMessage: Message = {
      id: messageId,
      content: inputText.trim(),
      role: 'user',
      timestamp: new Date(),
      status: isOnline ? 'sent' : 'pending',
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    
    // Scroll to bottom
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);

    try {
      if (isOnline) {
        // Send message directly
        setAgentTyping(true);
        const response = await APIClient.sendChatMessage({
          agentId,
          message: userMessage.content,
          messageId,
        });

        // Add agent response
        const agentMessage: Message = {
          id: `agent_${Date.now()}`,
          content: response.content,
          role: 'assistant',
          timestamp: new Date(),
          status: 'sent',
          agentType: response.agentType,
        };

        setMessages(prev => [...prev, agentMessage]);
      } else {
        // Queue for offline sync
        await SyncService.addPendingOperation({
          type: 'chat_message',
          data: {
            agentId,
            message: userMessage.content,
            messageId,
          },
          maxRetries: 3,
        });

        // Update message status
        setMessages(prev => 
          prev.map(msg => 
            msg.id === messageId 
              ? { ...msg, status: 'pending' }
              : msg
          )
        );
      }
    } catch (error) {
      Logger.error('ChatScreen', 'Failed to send message', error);
      
      // Update message status to failed
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'failed' }
            : msg
        )
      );

      Alert.alert(
        'Message Failed',
        'Failed to send message. It will be retried when connection is restored.',
        [{ text: 'OK' }]
      );
    } finally {
      setAgentTyping(false);
    }
  };

  const handleInfoPress = () => {
    Alert.alert(
      'Agent Info',
      `Agent: ${agentName}\nID: ${agentId}\nStatus: ${isOnline ? 'Online' : 'Offline'}`,
      [{ text: 'OK' }]
    );
  };

  const handleRetryMessage = async (messageId: string) => {
    const message = messages.find(m => m.id === messageId);
    if (!message || !isOnline) return;

    try {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'pending' }
            : msg
        )
      );

      const response = await APIClient.sendChatMessage({
        agentId,
        message: message.content,
        messageId,
      });

      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'sent' }
            : msg
        )
      );

      // Add agent response
      const agentMessage: Message = {
        id: `agent_${Date.now()}`,
        content: response.content,
        role: 'assistant',
        timestamp: new Date(),
        status: 'sent',
        agentType: response.agentType,
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      Logger.error('ChatScreen', 'Failed to retry message', error);
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'failed' }
            : msg
        )
      );
    }
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      {/* Connection Status Bar */}
      {!isOnline && (
        <View style={styles.offlineBar}>
          <Icon name="cloud-off" size={16} color="white" />
          <Text style={styles.offlineText}>Offline - Messages will sync when connected</Text>
        </View>
      )}

      {/* Messages */}
      <ScrollView
        ref={scrollViewRef}
        style={styles.messagesContainer}
        showsVerticalScrollIndicator={false}
        onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
      >
        <View style={styles.messagesContent}>
          {messages.length === 0 ? (
            <View style={styles.emptyState}>
              <Icon name="chat" size={64} color="#D1D5DB" />
              <Text style={styles.emptyStateText}>Start a conversation</Text>
              <Text style={styles.emptyStateSubtext}>
                Send a message to begin chatting with {agentName}
              </Text>
            </View>
          ) : (
            messages.map((message) => (
              <TouchableOpacity
                key={message.id}
                onPress={() => {
                  if (message.status === 'failed') {
                    Alert.alert(
                      'Retry Message',
                      'Would you like to retry sending this message?',
                      [
                        { text: 'Cancel', style: 'cancel' },
                        { text: 'Retry', onPress: () => handleRetryMessage(message.id) },
                      ]
                    );
                  }
                }}
                disabled={message.status !== 'failed'}
              >
                <MessageBubble message={message} />
              </TouchableOpacity>
            ))
          )}
          
          {/* Agent typing indicator */}
          {agentTyping && (
            <View style={styles.typingContainer}>
              <View style={styles.typingBubble}>
                <ActivityIndicator size="small" color="#6B7280" />
                <Text style={styles.typingText}>{agentName} is typing...</Text>
              </View>
            </View>
          )}
        </View>
      </ScrollView>

      {/* Input Area */}
      <View style={styles.inputContainer}>
        <View style={styles.inputWrapper}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder={`Message ${agentName}...`}
            multiline
            maxLength={1000}
            scrollEnabled
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!inputText.trim() || isLoading) && styles.sendButtonDisabled
            ]}
            onPress={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Icon name="send" size={20} color="white" />
            )}
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  offlineBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#EF4444',
    paddingVertical: 8,
    paddingHorizontal: 16,
    gap: 8,
  },
  offlineText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
  },
  messageContainer: {
    marginVertical: 4,
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  agentMessage: {
    alignItems: 'flex-start',
  },
  messageBubble: {
    maxWidth: width * 0.8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
  },
  userBubble: {
    backgroundColor: '#2563EB',
    borderBottomRightRadius: 4,
  },
  agentBubble: {
    backgroundColor: 'white',
    borderBottomLeftRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userText: {
    color: 'white',
  },
  agentText: {
    color: '#111827',
  },
  messageFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  timestamp: {
    fontSize: 12,
    opacity: 0.7,
  },
  userTimestamp: {
    color: 'white',
  },
  agentTimestamp: {
    color: '#6B7280',
  },
  statusIcon: {
    marginLeft: 4,
  },
  typingContainer: {
    alignItems: 'flex-start',
    marginVertical: 4,
  },
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
    borderBottomLeftRadius: 4,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  typingText: {
    fontSize: 14,
    color: '#6B7280',
    fontStyle: 'italic',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 64,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#9CA3AF',
    marginTop: 16,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#D1D5DB',
    marginTop: 8,
    textAlign: 'center',
  },
  inputContainer: {
    backgroundColor: 'white',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 12,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    maxHeight: 100,
    textAlignVertical: 'top',
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#2563EB',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#9CA3AF',
  },
});

export default ChatScreen;