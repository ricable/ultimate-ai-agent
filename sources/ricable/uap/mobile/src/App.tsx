// mobile/src/App.tsx
// Agent 22: React Native Mobile App for UAP Agents

import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { Platform, StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Screens
import AgentDashboard from './components/AgentDashboard';
import ChatScreen from './components/ChatScreen';
import OfflineScreen from './components/OfflineScreen';
import SettingsScreen from './components/SettingsScreen';
import ProfileScreen from './components/ProfileScreen';

// Services
import { initializeOfflineSync } from './services/OfflineSync';
import { initializePushNotifications } from './services/PushNotifications';
import { NetworkProvider } from './hooks/useNetwork';
import { AgentProvider } from './hooks/useAgents';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Main tab navigator
function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Dashboard':
              iconName = 'dashboard';
              break;
            case 'Chat':
              iconName = 'chat';
              break;
            case 'Offline':
              iconName = 'cloud-off';
              break;
            case 'Settings':
              iconName = 'settings';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#2563eb',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}
    >
      <Tab.Screen name="Dashboard" component={AgentDashboard} />
      <Tab.Screen name="Chat" component={ChatScreen} />
      <Tab.Screen name="Offline" component={OfflineScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

// Stack navigator for nested navigation
function RootNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Main" component={MainTabs} />
      <Stack.Screen name="Profile" component={ProfileScreen} />
    </Stack.Navigator>
  );
}

export default function App() {
  useEffect(() => {
    // Initialize mobile services
    const initServices = async () => {
      try {
        // Initialize offline sync capabilities
        await initializeOfflineSync();
        
        // Initialize push notifications
        await initializePushNotifications();
        
        console.log('Mobile services initialized successfully');
      } catch (error) {
        console.error('Failed to initialize mobile services:', error);
      }
    };

    initServices();
  }, []);

  return (
    <SafeAreaProvider>
      <NetworkProvider>
        <AgentProvider>
          <NavigationContainer>
            {Platform.OS === 'ios' && <StatusBar barStyle="dark-content" />}
            {Platform.OS === 'android' && (
              <StatusBar barStyle="dark-content" backgroundColor="#ffffff" />
            )}
            <RootNavigator />
          </NavigationContainer>
        </AgentProvider>
      </NetworkProvider>
    </SafeAreaProvider>
  );
}