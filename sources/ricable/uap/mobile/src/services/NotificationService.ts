import PushNotification from 'react-native-push-notification';
import { Platform, Alert, Linking, AppState } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Logger } from '../utils/Logger';

export interface NotificationData {
  id: string;
  title: string;
  message: string;
  data?: any;
  timestamp: number;
  type: 'chat_message' | 'sync_complete' | 'sync_failed' | 'agent_status' | 'system';
  priority: 'low' | 'normal' | 'high';
  actions?: NotificationAction[];
}

export interface NotificationAction {
  id: string;
  title: string;
  destructive?: boolean;
}

class NotificationServiceClass {
  private isInitialized = false;
  private hasPermission = false;
  private deviceToken: string | null = null;
  private notificationHistory: NotificationData[] = [];
  private listeners: Map<string, (notification: NotificationData) => void> = new Map();

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    Logger.info('NotificationService', 'Initializing notification service');

    try {
      // Configure push notifications
      PushNotification.configure({
        // Called when token is generated
        onRegister: (token) => {
          this.deviceToken = token.token;
          Logger.info('NotificationService', 'Device token registered', { token: token.token });
          this.saveDeviceToken(token.token);
        },

        // Called when a remote or local notification is opened or received
        onNotification: (notification) => {
          Logger.info('NotificationService', 'Notification received', notification);
          this.handleNotification(notification);
        },

        // Called when action is pressed
        onAction: (notification) => {
          Logger.info('NotificationService', 'Notification action pressed', notification);
          this.handleNotificationAction(notification);
        },

        // Called when the user fails to register for remote notifications
        onRegistrationError: (err) => {
          Logger.error('NotificationService', 'Failed to register for notifications', err);
        },

        // IOS ONLY: Called when a notification is delivered to a foreground app
        onRemoteNotification: (notification) => {
          Logger.info('NotificationService', 'Remote notification received', notification);
        },

        // Whether permissions should be requested on initialization
        requestPermissions: true,

        // Other settings
        popInitialNotification: true,
        permissions: {
          alert: true,
          badge: true,
          sound: true,
        },
      });

      // Create notification channels (Android)
      if (Platform.OS === 'android') {
        this.createNotificationChannels();
      }

      // Load notification history
      await this.loadNotificationHistory();

      // Check and request permissions
      await this.checkPermissions();

      this.isInitialized = true;
      Logger.info('NotificationService', 'Notification service initialized successfully');
    } catch (error) {
      Logger.error('NotificationService', 'Failed to initialize notification service', error);
      throw error;
    }
  }

  private createNotificationChannels(): void {
    const channels = [
      {
        channelId: 'chat_messages',
        channelName: 'Chat Messages',
        channelDescription: 'Notifications for new chat messages',
        importance: 4,
        soundName: 'default',
        vibrate: true,
      },
      {
        channelId: 'sync_notifications',
        channelName: 'Sync Notifications',
        channelDescription: 'Notifications for sync status updates',
        importance: 3,
        soundName: 'default',
        vibrate: false,
      },
      {
        channelId: 'agent_status',
        channelName: 'Agent Status',
        channelDescription: 'Notifications for agent status changes',
        importance: 2,
        soundName: 'default',
        vibrate: false,
      },
      {
        channelId: 'system_notifications',
        channelName: 'System Notifications',
        channelDescription: 'Important system notifications',
        importance: 4,
        soundName: 'default',
        vibrate: true,
      },
    ];

    channels.forEach(channel => {
      PushNotification.createChannel(
        {
          channelId: channel.channelId,
          channelName: channel.channelName,
          channelDescription: channel.channelDescription,
          playSound: true,
          soundName: channel.soundName,
          importance: channel.importance,
          vibrate: channel.vibrate,
        },
        (created) => {
          Logger.info('NotificationService', `Channel ${channel.channelId} created: ${created}`);
        }
      );
    });
  }

  async requestPermission(): Promise<boolean> {
    try {
      if (Platform.OS === 'ios') {
        // iOS permission request
        PushNotification.requestPermissions();
        
        // Check if permissions were granted
        return new Promise((resolve) => {
          PushNotification.checkPermissions((permissions) => {
            const hasPermission = permissions.alert && permissions.badge && permissions.sound;
            this.hasPermission = hasPermission;
            resolve(hasPermission);
          });
        });
      } else {
        // Android permissions are handled during configuration
        this.hasPermission = true;
        return true;
      }
    } catch (error) {
      Logger.error('NotificationService', 'Failed to request permissions', error);
      return false;
    }
  }

  async getPermissionStatus(): Promise<boolean> {
    if (Platform.OS === 'ios') {
      return new Promise((resolve) => {
        PushNotification.checkPermissions((permissions) => {
          const hasPermission = permissions.alert && permissions.badge && permissions.sound;
          this.hasPermission = hasPermission;
          resolve(hasPermission);
        });
      });
    } else {
      return this.hasPermission;
    }
  }

  async checkPermissions(): Promise<void> {
    const hasPermission = await this.getPermissionStatus();
    if (!hasPermission) {
      Logger.warn('NotificationService', 'Notification permissions not granted');
    }
  }

  async disable(): Promise<void> {
    try {
      // Clear all scheduled notifications
      PushNotification.cancelAllLocalNotifications();
      
      // On iOS, we can't truly disable notifications programmatically
      // User needs to do it in settings
      if (Platform.OS === 'ios') {
        Alert.alert(
          'Disable Notifications',
          'To disable notifications, please go to Settings > Notifications > UAP and turn off notifications.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Linking.openURL('app-settings:') },
          ]
        );
      }
      
      this.hasPermission = false;
      Logger.info('NotificationService', 'Notifications disabled');
    } catch (error) {
      Logger.error('NotificationService', 'Failed to disable notifications', error);
    }
  }

  async showNotification(data: Omit<NotificationData, 'id' | 'timestamp'>): Promise<string> {
    if (!this.hasPermission) {
      Logger.warn('NotificationService', 'Cannot show notification - no permission');
      return '';
    }

    const id = `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const timestamp = Date.now();
    
    const notification: NotificationData = {
      ...data,
      id,
      timestamp,
    };

    try {
      // Show local notification
      PushNotification.localNotification({
        id,
        title: data.title,
        message: data.message,
        channelId: this.getChannelId(data.type),
        priority: this.getPriority(data.priority),
        largeIcon: 'ic_launcher',
        smallIcon: 'ic_notification',
        userInfo: {
          ...data.data,
          notificationId: id,
          type: data.type,
        },
        actions: data.actions?.map(action => action.title) || [],
        invokeApp: true,
        autoCancel: true,
        vibrate: data.priority === 'high',
        playSound: true,
        soundName: 'default',
      });

      // Add to history
      this.notificationHistory.unshift(notification);
      
      // Keep only last 100 notifications
      if (this.notificationHistory.length > 100) {
        this.notificationHistory = this.notificationHistory.slice(0, 100);
      }
      
      await this.saveNotificationHistory();
      
      // Notify listeners
      this.listeners.forEach(listener => listener(notification));
      
      Logger.info('NotificationService', 'Notification shown', { id, title: data.title });
      return id;
    } catch (error) {
      Logger.error('NotificationService', 'Failed to show notification', error);
      return '';
    }
  }

  async cancelNotification(id: string): Promise<void> {
    try {
      PushNotification.cancelLocalNotifications({ id });
      Logger.info('NotificationService', 'Notification cancelled', { id });
    } catch (error) {
      Logger.error('NotificationService', 'Failed to cancel notification', error);
    }
  }

  async cancelAllNotifications(): Promise<void> {
    try {
      PushNotification.cancelAllLocalNotifications();
      Logger.info('NotificationService', 'All notifications cancelled');
    } catch (error) {
      Logger.error('NotificationService', 'Failed to cancel all notifications', error);
    }
  }

  addListener(id: string, callback: (notification: NotificationData) => void): void {
    this.listeners.set(id, callback);
  }

  removeListener(id: string): void {
    this.listeners.delete(id);
  }

  getNotificationHistory(): NotificationData[] {
    return [...this.notificationHistory];
  }

  async clearNotificationHistory(): Promise<void> {
    this.notificationHistory = [];
    await this.saveNotificationHistory();
    Logger.info('NotificationService', 'Notification history cleared');
  }

  getDeviceToken(): string | null {
    return this.deviceToken;
  }

  // Convenience methods for specific notification types
  async showChatMessage(agentName: string, message: string, data?: any): Promise<string> {
    return this.showNotification({
      title: `New message from ${agentName}`,
      message: message.length > 100 ? `${message.substring(0, 100)}...` : message,
      type: 'chat_message',
      priority: 'normal',
      data,
    });
  }

  async showSyncComplete(itemsCount: number): Promise<string> {
    return this.showNotification({
      title: 'Sync Complete',
      message: `Successfully synced ${itemsCount} items`,
      type: 'sync_complete',
      priority: 'low',
    });
  }

  async showSyncFailed(error: string): Promise<string> {
    return this.showNotification({
      title: 'Sync Failed',
      message: `Failed to sync data: ${error}`,
      type: 'sync_failed',
      priority: 'normal',
    });
  }

  async showAgentStatusChange(agentName: string, status: string): Promise<string> {
    return this.showNotification({
      title: 'Agent Status Changed',
      message: `${agentName} is now ${status}`,
      type: 'agent_status',
      priority: 'low',
    });
  }

  // Private methods

  private handleNotification(notification: any): void {
    // Handle notification tap/reception
    if (AppState.currentState === 'background' || AppState.currentState === 'inactive') {
      // App was opened from notification
      Logger.info('NotificationService', 'App opened from notification', notification);
    }
  }

  private handleNotificationAction(notification: any): void {
    // Handle notification action button press
    Logger.info('NotificationService', 'Notification action pressed', notification);
  }

  private getChannelId(type: NotificationData['type']): string {
    switch (type) {
      case 'chat_message':
        return 'chat_messages';
      case 'sync_complete':
      case 'sync_failed':
        return 'sync_notifications';
      case 'agent_status':
        return 'agent_status';
      case 'system':
        return 'system_notifications';
      default:
        return 'system_notifications';
    }
  }

  private getPriority(priority: NotificationData['priority']): 'low' | 'normal' | 'high' {
    return priority;
  }

  private async saveDeviceToken(token: string): Promise<void> {
    try {
      await AsyncStorage.setItem('device_token', token);
    } catch (error) {
      Logger.error('NotificationService', 'Failed to save device token', error);
    }
  }

  private async loadNotificationHistory(): Promise<void> {
    try {
      const history = await AsyncStorage.getItem('notification_history');
      if (history) {
        this.notificationHistory = JSON.parse(history);
      }
    } catch (error) {
      Logger.error('NotificationService', 'Failed to load notification history', error);
    }
  }

  private async saveNotificationHistory(): Promise<void> {
    try {
      await AsyncStorage.setItem('notification_history', JSON.stringify(this.notificationHistory));
    } catch (error) {
      Logger.error('NotificationService', 'Failed to save notification history', error);
    }
  }
}

export const NotificationService = new NotificationServiceClass();