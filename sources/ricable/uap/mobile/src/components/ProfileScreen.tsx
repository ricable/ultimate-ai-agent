import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Image,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useNavigation } from '@react-navigation/native';
import { APIClient } from '../services/APIClient';
import { OfflineService } from '../services/OfflineService';
import { Logger } from '../utils/Logger';

const { width } = Dimensions.get('window');

interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: string;
  joinDate: string;
  lastActive: string;
  preferences: {
    favoriteAgent: string;
    totalConversations: number;
    totalMessages: number;
  };
  stats: {
    sessionsThisWeek: number;
    averageSessionLength: string;
    mostActiveHour: string;
    totalUptime: string;
  };
}

const StatCard: React.FC<{ title: string; value: string; icon: string; color: string }> = ({ 
  title, 
  value, 
  icon, 
  color 
}) => {
  return (
    <View style={styles.statCard}>
      <View style={[styles.statIcon, { backgroundColor: color }]}>
        <Icon name={icon} size={20} color="white" />
      </View>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statTitle}>{title}</Text>
    </View>
  );
};

const ActionButton: React.FC<{ 
  title: string; 
  icon: string; 
  color: string; 
  onPress: () => void;
  description?: string;
}> = ({ title, icon, color, onPress, description }) => {
  return (
    <TouchableOpacity style={styles.actionButton} onPress={onPress}>
      <View style={styles.actionContent}>
        <View style={[styles.actionIcon, { backgroundColor: color }]}>
          <Icon name={icon} size={20} color="white" />
        </View>
        <View style={styles.actionText}>
          <Text style={styles.actionTitle}>{title}</Text>
          {description && (
            <Text style={styles.actionDescription}>{description}</Text>
          )}
        </View>
        <Icon name="chevron-right" size={20} color="#9CA3AF" />
      </View>
    </TouchableOpacity>
  );
};

const ProfileScreen: React.FC = () => {
  const navigation = useNavigation();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      // Try to load from API first
      const profileData = await APIClient.getUserProfile();
      setProfile(profileData);
    } catch (error) {
      Logger.error('ProfileScreen', 'Failed to load profile from API', error);
      
      // Fallback to cached profile or demo data
      const cachedProfile = await OfflineService.getCachedProfile();
      if (cachedProfile) {
        setProfile(cachedProfile);
      } else {
        // Demo profile data
        setProfile({
          id: 'demo-user',
          name: 'Demo User',
          email: 'demo@uap.ai',
          role: 'User',
          joinDate: '2024-01-01',
          lastActive: new Date().toISOString(),
          preferences: {
            favoriteAgent: 'CopilotKit',
            totalConversations: 42,
            totalMessages: 156,
          },
          stats: {
            sessionsThisWeek: 8,
            averageSessionLength: '12 min',
            mostActiveHour: '2 PM',
            totalUptime: '3h 24m',
          },
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadProfile();
    setRefreshing(false);
  };

  const handleEditProfile = () => {
    Alert.alert(
      'Edit Profile',
      'Profile editing is not yet implemented in this demo version.',
      [{ text: 'OK' }]
    );
  };

  const handleChangePassword = () => {
    Alert.alert(
      'Change Password',
      'Password change is not yet implemented in this demo version.',
      [{ text: 'OK' }]
    );
  };

  const handlePrivacySettings = () => {
    Alert.alert(
      'Privacy Settings',
      'Privacy settings management is not yet implemented in this demo version.',
      [{ text: 'OK' }]
    );
  };

  const handleDataExport = () => {
    Alert.alert(
      'Export Data',
      'Would you like to export your profile data and conversation history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Export',
          onPress: async () => {
            try {
              // Implementation would export user data
              Alert.alert('Success', 'Data export started. You will receive a download link shortly.');
            } catch (error) {
              Logger.error('ProfileScreen', 'Failed to export data', error);
              Alert.alert('Error', 'Failed to export data');
            }
          }
        }
      ]
    );
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'Are you sure you want to delete your account? This action cannot be undone and will permanently delete all your data.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            Alert.alert(
              'Confirm Deletion',
              'Please type "DELETE" to confirm account deletion.',
              [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'I understand',
                  style: 'destructive',
                  onPress: async () => {
                    try {
                      // Implementation would delete account
                      Alert.alert('Account Deleted', 'Your account has been deleted.');
                    } catch (error) {
                      Logger.error('ProfileScreen', 'Failed to delete account', error);
                      Alert.alert('Error', 'Failed to delete account');
                    }
                  }
                }
              ]
            );
          }
        }
      ]
    );
  };

  const handleLogout = () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          onPress: async () => {
            try {
              await APIClient.logout();
              // Navigation to login screen would happen here
              Alert.alert('Logged Out', 'You have been successfully logged out.');
            } catch (error) {
              Logger.error('ProfileScreen', 'Failed to logout', error);
              Alert.alert('Error', 'Failed to logout');
            }
          }
        }
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator large color="#2563EB" />
        <Text style={styles.loadingText}>Loading profile...</Text>
      </View>
    );
  }

  if (!profile) {
    return (
      <View style={styles.errorContainer}>
        <Icon name="error" size={48} color="#EF4444" />
        <Text style={styles.errorText}>Failed to load profile</Text>
        <TouchableOpacity style={styles.retryButton} onPress={loadProfile}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton} 
          onPress={() => navigation.goBack()}
        >
          <Icon name="arrow-back" size={24} color="#111827" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Profile</Text>
        <TouchableOpacity style={styles.editButton} onPress={handleEditProfile}>
          <Icon name="edit" size={24} color="#2563EB" />
        </TouchableOpacity>
      </View>

      {/* Profile Info */}
      <View style={styles.profileSection}>
        <View style={styles.avatarContainer}>
          {profile.avatar ? (
            <Image source={{ uri: profile.avatar }} style={styles.avatar} />
          ) : (
            <View style={styles.avatarPlaceholder}>
              <Icon name="person" size={48} color="#9CA3AF" />
            </View>
          )}
        </View>
        <Text style={styles.profileName}>{profile.name}</Text>
        <Text style={styles.profileEmail}>{profile.email}</Text>
        <View style={styles.profileMeta}>
          <View style={styles.metaItem}>
            <Icon name="badge" size={16} color="#6B7280" />
            <Text style={styles.metaText}>{profile.role}</Text>
          </View>
          <View style={styles.metaItem}>
            <Icon name="calendar-today" size={16} color="#6B7280" />
            <Text style={styles.metaText}>Joined {new Date(profile.joinDate).toLocaleDateString()}</Text>
          </View>
        </View>
      </View>

      {/* Stats */}
      <View style={styles.statsSection}>
        <Text style={styles.sectionTitle}>Usage Statistics</Text>
        <View style={styles.statsGrid}>
          <StatCard
            title="This Week"
            value={profile.stats.sessionsThisWeek.toString()}
            icon="trending-up"
            color="#2563EB"
          />
          <StatCard
            title="Avg Session"
            value={profile.stats.averageSessionLength}
            icon="schedule"
            color="#059669"
          />
          <StatCard
            title="Most Active"
            value={profile.stats.mostActiveHour}
            icon="access-time"
            color="#F59E0B"
          />
          <StatCard
            title="Total Time"
            value={profile.stats.totalUptime}
            icon="timer"
            color="#8B5CF6"
          />
        </View>
      </View>

      {/* Preferences */}
      <View style={styles.preferencesSection}>
        <Text style={styles.sectionTitle}>Preferences</Text>
        <View style={styles.preferenceCard}>
          <View style={styles.preferenceItem}>
            <Icon name="favorite" size={20} color="#EF4444" />
            <Text style={styles.preferenceLabel}>Favorite Agent</Text>
            <Text style={styles.preferenceValue}>{profile.preferences.favoriteAgent}</Text>
          </View>
          <View style={styles.preferenceItem}>
            <Icon name="chat" size={20} color="#2563EB" />
            <Text style={styles.preferenceLabel}>Total Conversations</Text>
            <Text style={styles.preferenceValue}>{profile.preferences.totalConversations}</Text>
          </View>
          <View style={styles.preferenceItem}>
            <Icon name="message" size={20} color="#059669" />
            <Text style={styles.preferenceLabel}>Total Messages</Text>
            <Text style={styles.preferenceValue}>{profile.preferences.totalMessages}</Text>
          </View>
        </View>
      </View>

      {/* Actions */}
      <View style={styles.actionsSection}>
        <Text style={styles.sectionTitle}>Account Actions</Text>
        <View style={styles.actionsContainer}>
          <ActionButton
            title="Change Password"
            icon="lock"
            color="#2563EB"
            onPress={handleChangePassword}
            description="Update your account password"
          />
          <ActionButton
            title="Privacy Settings"
            icon="privacy-tip"
            color="#059669"
            onPress={handlePrivacySettings}
            description="Manage your privacy preferences"
          />
          <ActionButton
            title="Export Data"
            icon="file-download"
            color="#F59E0B"
            onPress={handleDataExport}
            description="Download your data and conversations"
          />
          <ActionButton
            title="Logout"
            icon="logout"
            color="#6B7280"
            onPress={handleLogout}
            description="Sign out of your account"
          />
          <ActionButton
            title="Delete Account"
            icon="delete-forever"
            color="#EF4444"
            onPress={handleDeleteAccount}
            description="Permanently delete your account"
          />
        </View>
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
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#6B7280',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F9FAFB',
    padding: 20,
  },
  errorText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: '600',
    color: '#EF4444',
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 16,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#2563EB',
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: 'white',
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
  },
  editButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#EBF4FF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileSection: {
    backgroundColor: 'white',
    alignItems: 'center',
    paddingVertical: 32,
    marginBottom: 24,
  },
  avatarContainer: {
    marginBottom: 16,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  avatarPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#F3F4F6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: 16,
    color: '#6B7280',
    marginBottom: 16,
  },
  profileMeta: {
    flexDirection: 'row',
    gap: 24,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaText: {
    fontSize: 14,
    color: '#6B7280',
  },
  statsSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 48) / 2,
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#111827',
  },
  statTitle: {
    fontSize: 12,
    color: '#6B7280',
    textAlign: 'center',
    marginTop: 4,
  },
  preferencesSection: {
    marginBottom: 24,
  },
  preferenceCard: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    borderRadius: 12,
    padding: 16,
  },
  preferenceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  preferenceLabel: {
    flex: 1,
    fontSize: 16,
    color: '#374151',
    marginLeft: 12,
  },
  preferenceValue: {
    fontSize: 16,
    fontWeight: '500',
    color: '#111827',
  },
  actionsSection: {
    marginBottom: 32,
  },
  actionsContainer: {
    backgroundColor: 'white',
    marginHorizontal: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  actionButton: {
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  actionContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  actionIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  actionText: {
    flex: 1,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#111827',
  },
  actionDescription: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 2,
  },
});

export default ProfileScreen;