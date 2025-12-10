/**
 * Provider Communication
 * Secure messaging between providers and patients
 */

import { ProviderCommunication } from './types';

export class ProviderCommunicationService {
  private messages: Map<string, ProviderCommunication[]>;
  private unreadMessages: Map<string, Set<string>>; // userId -> messageIds

  constructor() {
    this.messages = new Map();
    this.unreadMessages = new Map();
  }

  /**
   * Send message from provider to patient or vice versa
   */
  async sendMessage(message: Omit<ProviderCommunication, 'id' | 'sentAt'>): Promise<ProviderCommunication> {
    const communication: ProviderCommunication = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      ...message,
      sentAt: new Date()
    };

    // Store message by query
    const queryMessages = this.messages.get(message.queryId) || [];
    queryMessages.push(communication);
    this.messages.set(message.queryId, queryMessages);

    // Mark as unread for recipient
    const unread = this.unreadMessages.get(message.to) || new Set();
    unread.add(communication.id);
    this.unreadMessages.set(message.to, unread);

    console.log(`[COMMUNICATION] Message sent from ${message.from} to ${message.to} for query ${message.queryId}`);

    return communication;
  }

  /**
   * Get messages for query
   */
  getMessagesForQuery(queryId: string): ProviderCommunication[] {
    return this.messages.get(queryId) || [];
  }

  /**
   * Get conversation between provider and patient
   */
  getConversation(queryId: string, userId1: string, userId2: string): ProviderCommunication[] {
    const queryMessages = this.messages.get(queryId) || [];

    return queryMessages.filter(msg =>
      (msg.from === userId1 && msg.to === userId2) ||
      (msg.from === userId2 && msg.to === userId1)
    ).sort((a, b) => a.sentAt.getTime() - b.sentAt.getTime());
  }

  /**
   * Mark message as read
   */
  async markAsRead(messageId: string, userId: string): Promise<boolean> {
    // Find message
    let foundMessage: ProviderCommunication | undefined;
    for (const messages of this.messages.values()) {
      foundMessage = messages.find(m => m.id === messageId);
      if (foundMessage) break;
    }

    if (!foundMessage || foundMessage.to !== userId) {
      return false;
    }

    foundMessage.readAt = new Date();

    // Remove from unread
    const unread = this.unreadMessages.get(userId);
    if (unread) {
      unread.delete(messageId);
    }

    return true;
  }

  /**
   * Get unread messages for user
   */
  getUnreadMessages(userId: string): ProviderCommunication[] {
    const unreadIds = this.unreadMessages.get(userId);
    if (!unreadIds) {
      return [];
    }

    const unreadMessages: ProviderCommunication[] = [];

    for (const messages of this.messages.values()) {
      for (const message of messages) {
        if (unreadIds.has(message.id)) {
          unreadMessages.push(message);
        }
      }
    }

    return unreadMessages.sort((a, b) => b.sentAt.getTime() - a.sentAt.getTime());
  }

  /**
   * Get unread count for user
   */
  getUnreadCount(userId: string): number {
    const unread = this.unreadMessages.get(userId);
    return unread ? unread.size : 0;
  }

  /**
   * Mark all messages as read for user in query
   */
  async markAllAsReadForQuery(queryId: string, userId: string): Promise<number> {
    const queryMessages = this.messages.get(queryId) || [];
    let count = 0;

    for (const message of queryMessages) {
      if (message.to === userId && !message.readAt) {
        message.readAt = new Date();

        const unread = this.unreadMessages.get(userId);
        if (unread) {
          unread.delete(message.id);
        }

        count++;
      }
    }

    return count;
  }

  /**
   * Send secure file attachment
   */
  async sendAttachment(
    queryId: string,
    from: string,
    to: string,
    fileName: string,
    fileUrl: string,
    message?: string
  ): Promise<ProviderCommunication> {
    return this.sendMessage({
      from,
      to,
      queryId,
      message: message || `Sent attachment: ${fileName}`,
      attachments: [fileUrl],
      encrypted: true
    });
  }

  /**
   * Get message statistics
   */
  getStats(): {
    totalMessages: number;
    totalConversations: number;
    unreadTotal: number;
    encryptedMessages: number;
  } {
    let totalMessages = 0;
    let encryptedMessages = 0;
    let unreadTotal = 0;

    for (const messages of this.messages.values()) {
      totalMessages += messages.length;
      encryptedMessages += messages.filter(m => m.encrypted).length;
    }

    for (const unread of this.unreadMessages.values()) {
      unreadTotal += unread.size;
    }

    return {
      totalMessages,
      totalConversations: this.messages.size,
      unreadTotal,
      encryptedMessages
    };
  }
}
