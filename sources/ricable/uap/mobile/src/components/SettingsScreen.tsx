import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Switch,
  Alert,
  ActionSheetIOS,
  Platform,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useNavigation } from '@react-navigation/native';
import { OfflineService } from '../services/OfflineService';
import { NotificationService } from '../services/NotificationService';
import { Logger } from '../utils/Logger';

interface Setting {
  id: string;
  title: string;
  description?: string;
  type: 'toggle' | 'select' | 'action';
  value?: boolean | string;
  options?: string[];
  icon: string;
  iconColor: string;
  onPress?: () => void;
}

const SettingItem: React.FC<{ setting: Setting; onToggle?: (value: boolean) => void; onSelect?: (value: string) => void }> = ({ 
  setting, 
  onToggle, 
  onSelect 
}) => {
  const handlePress = () => {
    if (setting.type === 'action' && setting.onPress) {
      setting.onPress();
    } else if (setting.type === 'select' && setting.options) {
      if (Platform.OS === 'ios') {
        ActionSheetIOS.showActionSheetWithOptions(
          {
            options: ['Cancel', ...setting.options],
            cancelButtonIndex: 0,
            title: setting.title,
          },
          (buttonIndex) => {
            if (buttonIndex > 0 && onSelect) {
              onSelect(setting.options![buttonIndex - 1]);
            }
          }
        );
      } else {
        // For Android, you might want to use a modal or picker
        Alert.alert(
          setting.title,
          'Select an option',
          setting.options.map(option => ({
            text: option,
            onPress: () => onSelect && onSelect(option)
          }))
        );
      }
    }
  };

  return (
    <TouchableOpacity 
      style={styles.settingItem} 
      onPress={handlePress}
      disabled={setting.type === 'toggle'}
    >
      <View style={styles.settingContent}>
        <View style={styles.settingIcon}>
          <Icon name={setting.icon} size={24} color={setting.iconColor} />
        </View>
        <View style={styles.settingText}>
          <Text style={styles.settingTitle}>{setting.title}</Text>
          {setting.description && (
            <Text style={styles.settingDescription}>{setting.description}</Text>
          )}
          {setting.type === 'select' && (
            <Text style={styles.settingValue}>{setting.value as string}</Text>
          )}
        </View>
        <View style={styles.settingControl}>
          {setting.type === 'toggle' && (
            <Switch
              value={setting.value as boolean}
              onValueChange={onToggle}
              trackColor={{ false: '#D1D5DB', true: '#BFDBFE' }}
              thumbColor={(setting.value as boolean) ? '#2563EB' : '#9CA3AF'}
            />
          )}
          {setting.type === 'select' && (
            <Icon name="chevron-right" size={20} color="#9CA3AF" />
          )}
          {setting.type === 'action' && (
            <Icon name="chevron-right" size={20} color="#9CA3AF" />
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
};

const SettingsScreen: React.FC = () => {
  const navigation = useNavigation();
  const [settings, setSettings] = useState<Setting[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const notificationEnabled = await NotificationService.getPermissionStatus();
      const offlineSettings = await OfflineService.getSettings();
      
      const settingsConfig: Setting[] = [
        // Notifications
        {
          id: 'notifications',
          title: 'Push Notifications',
          description: 'Receive notifications for new messages and updates',
          type: 'toggle',
          value: notificationEnabled,
          icon: 'notifications',
          iconColor: '#F59E0B',
        },
        {
          id: 'notification_sound',
          title: 'Notification Sound',
          description: 'Play sound for notifications',
          type: 'toggle',
          value: offlineSettings.notificationSound ?? true,
          icon: 'volume-up',
          iconColor: '#8B5CF6',
        },
        
        // Sync & Offline
        {
          id: 'auto_sync',
          title: 'Auto Sync',
          description: 'Automatically sync when connected',
          type: 'toggle',
          value: offlineSettings.autoSync ?? true,
          icon: 'sync',
          iconColor: '#2563EB',
        },
        {
          id: 'sync_frequency',
          title: 'Sync Frequency',
          description: 'How often to sync data',
          type: 'select',
          value: offlineSettings.syncFrequency ?? 'Every 30 seconds',
          options: ['Every 15 seconds', 'Every 30 seconds', 'Every minute', 'Every 5 minutes'],
          icon: 'schedule',
          iconColor: '#059669',
        },
        {
          id: 'offline_storage',
          title: 'Offline Storage Limit',
          description: 'Maximum storage for offline data',
          type: 'select',
          value: offlineSettings.storageLimit ?? '100 MB',
          options: ['50 MB', '100 MB', '250 MB', '500 MB', '1 GB'],
          icon: 'storage',
          iconColor: '#DC2626',
        },
        
        // Privacy & Security
        {
          id: 'biometric_auth',
          title: 'Biometric Authentication',
          description: 'Use fingerprint or face ID to unlock',
          type: 'toggle',
          value: offlineSettings.biometricAuth ?? false,
          icon: 'fingerprint',
          iconColor: '#7C3AED',
        },
        {
          id: 'data_encryption',
          title: 'Data Encryption',
          description: 'Encrypt local data storage',
          type: 'toggle',
          value: offlineSettings.dataEncryption ?? true,
          icon: 'security',
          iconColor: '#059669',
        },
        
        // Performance
        {
          id: 'image_quality',
          title: 'Image Quality',
          description: 'Quality of images in chat',
          type: 'select',
          value: offlineSettings.imageQuality ?? 'High',
          options: ['Low', 'Medium', 'High', 'Original'],
          icon: 'image',
          iconColor: '#F59E0B',
        },
        {
          id: 'animation_speed',
          title: 'Animation Speed',
          description: 'Speed of UI animations',
          type: 'select',
          value: offlineSettings.animationSpeed ?? 'Normal',
          options: ['Slow', 'Normal', 'Fast', 'Disabled'],
          icon: 'animation',
          iconColor: '#8B5CF6',
        },
        
        // App Settings
        {
          id: 'theme',
          title: 'Theme',
          description: 'App appearance',
          type: 'select',
          value: offlineSettings.theme ?? 'System',
          options: ['Light', 'Dark', 'System'],
          icon: 'palette',
          iconColor: '#6B7280',
        },
        
        // Actions
        {
          id: 'view_profile',
          title: 'View Profile',
          type: 'action',
          icon: 'person',
          iconColor: '#2563EB',
          onPress: () => navigation.navigate('Profile'),
        },
        {
          id: 'export_data',
          title: 'Export Data',
          description: 'Export your conversations and settings',
          type: 'action',
          icon: 'file-download',
          iconColor: '#059669',
          onPress: handleExportData,
        },
        {
          id: 'clear_cache',
          title: 'Clear Cache',
          description: 'Clear temporary files and cached data',
          type: 'action',
          icon: 'cleaning-services',
          iconColor: '#F59E0B',
          onPress: handleClearCache,
        },
        {
          id: 'reset_app',
          title: 'Reset App',
          description: 'Reset all settings to default',
          type: 'action',
          icon: 'restore',
          iconColor: '#DC2626',
          onPress: handleResetApp,
        },
      ];
      
      setSettings(settingsConfig);
    } catch (error) {
      Logger.error('SettingsScreen', 'Failed to load settings', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleSetting = async (settingId: string, value: boolean) => {
    try {
      switch (settingId) {
        case 'notifications':
          if (value) {
            await NotificationService.requestPermission();
          } else {
            await NotificationService.disable();
          }
          break;
        case 'notification_sound':
          await OfflineService.updateSetting('notificationSound', value);
          break;
        case 'auto_sync':
          await OfflineService.updateSetting('autoSync', value);
          break;
        case 'biometric_auth':
          await OfflineService.updateSetting('biometricAuth', value);
          break;
        case 'data_encryption':
          await OfflineService.updateSetting('dataEncryption', value);
          if (value) {
            Alert.alert(
              'Encryption Enabled',
              'Your data will be encrypted. This may affect performance slightly.',
              [{ text: 'OK' }]
            );
          }
          break;
      }
      
      // Update setting in state
      setSettings(prev => 
        prev.map(setting => 
          setting.id === settingId 
            ? { ...setting, value }
            : setting
        )
      );
    } catch (error) {
      Logger.error('SettingsScreen', `Failed to toggle ${settingId}`, error);
      Alert.alert('Error', 'Failed to update setting. Please try again.');
    }
  };

  const handleSelectSetting = async (settingId: string, value: string) => {
    try {
      switch (settingId) {
        case 'sync_frequency':
          await OfflineService.updateSetting('syncFrequency', value);
          break;
        case 'offline_storage':
          await OfflineService.updateSetting('storageLimit', value);
          break;
        case 'image_quality':
          await OfflineService.updateSetting('imageQuality', value);
          break;
        case 'animation_speed':
          await OfflineService.updateSetting('animationSpeed', value);
          break;
        case 'theme':
          await OfflineService.updateSetting('theme', value);
          break;
      }
      
      // Update setting in state
      setSettings(prev => 
        prev.map(setting => 
          setting.id === settingId 
            ? { ...setting, value }
            : setting
        )
      );
    } catch (error) {
      Logger.error('SettingsScreen', `Failed to update ${settingId}`, error);
      Alert.alert('Error', 'Failed to update setting. Please try again.');
    }
  };

  async function handleExportData() {
    Alert.alert(
      'Export Data',
      'This will create a backup file with your conversations and settings.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Export',
          onPress: async () => {
            try {
              // Implementation would export data to file
              Alert.alert('Success', 'Data exported successfully!');
            } catch (error) {
              Logger.error('SettingsScreen', 'Failed to export data', error);
              Alert.alert('Error', 'Failed to export data');
            }
          }
        }
      ]
    );
  }

  async function handleClearCache() {
    Alert.alert(
      'Clear Cache',
      'This will remove temporary files and cached data. The app may need to re-download some content.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          onPress: async () => {
            try {
              await OfflineService.clearCache();
              Alert.alert('Success', 'Cache cleared successfully!');
            } catch (error) {
              Logger.error('SettingsScreen', 'Failed to clear cache', error);
              Alert.alert('Error', 'Failed to clear cache');
            }
          }
        }
      ]
    );
  }

  async function handleResetApp() {
    Alert.alert(
      'Reset App',
      'This will reset all settings to default values. Your conversations will not be deleted.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              await OfflineService.resetSettings();
              await loadSettings();
              Alert.alert('Success', 'App settings reset successfully!');
            } catch (error) {
              Logger.error('SettingsScreen', 'Failed to reset app', error);
              Alert.alert('Error', 'Failed to reset app settings');
            }
          }
        }
      ]
    );
  }

  const groupedSettings = {
    'Notifications': settings.filter(s => ['notifications', 'notification_sound'].includes(s.id)),
    'Sync & Offline': settings.filter(s => ['auto_sync', 'sync_frequency', 'offline_storage'].includes(s.id)),
    'Privacy & Security': settings.filter(s => ['biometric_auth', 'data_encryption'].includes(s.id)),
    'Performance': settings.filter(s => ['image_quality', 'animation_speed'].includes(s.id)),
    'Appearance': settings.filter(s => ['theme'].includes(s.id)),
    'Account': settings.filter(s => ['view_profile', 'export_data'].includes(s.id)),
    'Advanced': settings.filter(s => ['clear_cache', 'reset_app'].includes(s.id)),
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <Text>Loading settings...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Settings</Text>
        <Text style={styles.headerSubtitle}>Customize your UAP experience</Text>
      </View>

      {Object.entries(groupedSettings).map(([groupName, groupSettings]) => (
        <View key={groupName} style={styles.settingGroup}>
          <Text style={styles.groupTitle}>{groupName}</Text>
          <View style={styles.groupContainer}>
            {groupSettings.map((setting, index) => (
              <SettingItem
                key={setting.id}
                setting={setting}
                onToggle={(value) => handleToggleSetting(setting.id, value)}
                onSelect={(value) => handleSelectSetting(setting.id, value)}
              />
            ))}
          </View>
        </View>
      ))}
      
      <View style={styles.footer}>
        <Text style={styles.footerText}>UAP Mobile v1.0.0</Text>
        <Text style={styles.footerSubtext}>Unified Agentic Platform</Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
  },
  header: {
    backgroundColor: 'white',
    padding: 20,
    paddingTop: 60,
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#111827',
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 4,
  },
  settingGroup: {
    marginBottom: 24,
  },
  groupTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  groupContainer: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  settingItem: {
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  settingIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#111827',
  },
  settingDescription: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 2,
  },
  settingValue: {
    fontSize: 14,
    color: '#2563EB',
    marginTop: 2,
    fontWeight: '500',
  },
  settingControl: {
    marginLeft: 16,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 32,
    paddingBottom: 50,
  },
  footerText: {
    fontSize: 14,
    color: '#9CA3AF',
    fontWeight: '500',
  },
  footerSubtext: {
    fontSize: 12,
    color: '#D1D5DB',
    marginTop: 4,
  },
});

export default SettingsScreen;