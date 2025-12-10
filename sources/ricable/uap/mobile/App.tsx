import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import { Provider as ReduxProvider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { Alert, AppState, AppStateStatus } from 'react-native';
import NetInfo from '@react-native-community/netinfo';
import Icon from 'react-native-vector-icons/MaterialIcons';

import { store, persistor } from './src/store';
import { AgentScreen } from './src/screens/AgentScreen';
import { ChatScreen } from './src/screens/ChatScreen';
import { SettingsScreen } from './src/screens/SettingsScreen';
import { DocumentScreen } from './src/screens/DocumentScreen';
import { LoadingScreen } from './src/components/LoadingScreen';
import { SyncService } from './src/services/SyncService';
import { NotificationService } from './src/services/NotificationService';
import { OfflineService } from './src/services/OfflineService';
import { theme } from './src/theme';

const Tab = createBottomTabNavigator();

const App: React.FC = () => {
  useEffect(() => {
    // Initialize services
    NotificationService.initialize();
    OfflineService.initialize();
    SyncService.initialize();

    // Monitor network status
    const unsubscribe = NetInfo.addEventListener(state => {
      if (state.isConnected) {
        SyncService.syncPendingOperations();
      }
    });

    // Handle app state changes for background sync
    const handleAppStateChange = (nextAppState: AppStateStatus) => {
      if (nextAppState === 'background') {
        SyncService.scheduleBackgroundSync();
      } else if (nextAppState === 'active') {
        SyncService.resumeSync();
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);

    return () => {
      unsubscribe();
      subscription?.remove();
    };
  }, []);

  return (
    <ReduxProvider store={store}>
      <PersistGate loading={<LoadingScreen />} persistor={persistor}>
        <PaperProvider theme={theme}>
          <NavigationContainer>
            <Tab.Navigator
              screenOptions={({ route }) => ({
                tabBarIcon: ({ focused, color, size }) => {
                  let iconName: string;

                  switch (route.name) {
                    case 'Agents':
                      iconName = 'smart-toy';
                      break;
                    case 'Chat':
                      iconName = 'chat';
                      break;
                    case 'Documents':
                      iconName = 'description';
                      break;
                    case 'Settings':
                      iconName = 'settings';
                      break;
                    default:
                      iconName = 'help';
                  }

                  return <Icon name={iconName} size={size} color={color} />;
                },
                tabBarActiveTintColor: theme.colors.primary,
                tabBarInactiveTintColor: 'gray',
                headerShown: false,
              })}
            >
              <Tab.Screen name="Agents" component={AgentScreen} />
              <Tab.Screen name="Chat" component={ChatScreen} />
              <Tab.Screen name="Documents" component={DocumentScreen} />
              <Tab.Screen name="Settings" component={SettingsScreen} />
            </Tab.Navigator>
          </NavigationContainer>
          <StatusBar style="auto" />
        </PaperProvider>
      </PersistGate>
    </ReduxProvider>
  );
};

export default App;