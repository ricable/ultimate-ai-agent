# UAP Mobile App

React Native mobile application for the UAP (Unified Agentic Platform) with offline-first capabilities and edge computing integration.

## Features

### Core Features
- **Cross-platform**: iOS and Android support via React Native and Expo
- **Offline-first**: Complete offline functionality with intelligent sync
- **Real-time sync**: WebSocket-based synchronization with backend
- **Push notifications**: Real-time notifications for agent responses
- **Edge computing**: Integration with WebAssembly edge runtime
- **Secure authentication**: JWT-based authentication with biometric support

### Agent Interaction
- Multi-agent chat interface
- Voice input and text-to-speech
- Document upload and processing
- Agent marketplace and discovery
- Conversation history and search

### Offline Capabilities
- Offline conversation storage
- Offline document processing
- Background sync when connectivity restored
- Conflict resolution for concurrent edits
- Progressive data sync with priority queuing

### Performance & UX
- Optimized for mobile devices
- Smooth animations and transitions
- Dark mode and accessibility support
- Multi-language support
- Responsive design for tablets

## Architecture

### Technology Stack
- **Framework**: React Native 0.73 with Expo 50
- **Language**: TypeScript
- **State Management**: Redux Toolkit with Redux Persist
- **UI Components**: React Native Paper (Material Design)
- **Navigation**: React Navigation 6
- **Offline Storage**: AsyncStorage with SQLite for complex data
- **Networking**: WebSocket for real-time, HTTP for REST APIs
- **Notifications**: Expo Notifications with Firebase

### App Structure
```
src/
├── components/          # Reusable UI components
│   ├── agents/         # Agent-specific components
│   ├── chat/           # Chat interface components
│   ├── common/         # Common UI components
│   └── offline/        # Offline status components
├── screens/            # Main app screens
│   ├── AgentScreen.tsx
│   ├── ChatScreen.tsx
│   ├── DocumentScreen.tsx
│   └── SettingsScreen.tsx
├── services/           # Business logic services
│   ├── APIClient.ts    # HTTP API client
│   ├── SyncService.ts  # Offline sync management
│   ├── NotificationService.ts
│   └── OfflineService.ts
├── store/              # Redux state management
│   ├── slices/         # Redux slices
│   └── index.ts        # Store configuration
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
└── theme/              # App theming
```

## Development Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Expo CLI: `npm install -g @expo/cli`
- iOS: Xcode 14+ and iOS Simulator
- Android: Android Studio and Android Emulator

### Installation

1. **Install dependencies**:
   ```bash
   cd mobile
   npm install
   ```

2. **Configure environment**:
   Create `.env` file with:
   ```
   EXPO_PUBLIC_API_URL=http://localhost:8000
   EXPO_PUBLIC_WS_URL=ws://localhost:8000
   EXPO_PUBLIC_EDGE_URL=http://localhost:3001
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

### Running on Devices

**iOS Simulator**:
```bash
npm run ios
```

**Android Emulator**:
```bash
npm run android
```

**Physical Device**:
1. Install Expo Go app
2. Scan QR code from terminal
3. App will load on device

## Key Services

### SyncService
Manages offline-online synchronization:
- Queues operations when offline
- Batch syncs when online
- Handles sync conflicts
- Background sync with priority

```typescript
// Queue an offline operation
await SyncService.addPendingOperation({
  type: 'chat_message',
  data: messageData,
  maxRetries: 3
});

// Trigger manual sync
await SyncService.syncNow();
```

### OfflineService
Manages offline data storage:
- Local data persistence
- Version control for data
- Cleanup of old data
- Storage size management

```typescript
// Store data offline
await OfflineService.storeData('conversation', conversationId, data);

// Get offline data
const data = await OfflineService.getData('conversation', conversationId);

// Check if online
const isOnline = await OfflineService.isOnline();
```

### NotificationService
Manages push notifications:
- Local and remote notifications
- Background notification handling
- Notification scheduling
- Badge management

```typescript
// Show local notification
await NotificationService.showLocalNotification(
  'Agent Response',
  'Your AI assistant has responded',
  { type: 'agent_message', agentId: 'agent-1' }
);

// Schedule notification
const id = await NotificationService.scheduleNotification(
  'Reminder',
  'Check your agents',
  { seconds: 3600 } // 1 hour
);
```

## State Management

### Redux Store Structure
```typescript
interface RootState {
  agents: AgentsState;        // Agent list and selection
  chat: ChatState;            // Chat conversations
  documents: DocumentsState;  // Document management
  sync: SyncState;           // Sync status and queue
  auth: AuthState;           // Authentication state
  offline: OfflineState;     // Offline status and data
}
```

### Key Redux Slices

**AgentsSlice**: Manages agent state
```typescript
const { agents, loading, selectedAgent } = useSelector((state: RootState) => state.agents);
dispatch(fetchAgents());
dispatch(selectAgent(agentId));
```

**ChatSlice**: Manages conversations
```typescript
const { conversations, messages } = useSelector((state: RootState) => state.chat);
dispatch(sendMessage({ conversationId, content }));
dispatch(addMessage(message));
```

**OfflineSlice**: Manages offline state
```typescript
const { isOnline, pendingOperations } = useSelector((state: RootState) => state.offline);
dispatch(setOnlineStatus(true));
dispatch(addPendingOperation(operation));
```

## Offline-First Design

### Data Flow
1. **User Action**: User interacts with app
2. **Local Storage**: Data saved locally immediately
3. **UI Update**: Interface updates optimistically
4. **Queue Operation**: Sync operation queued if offline
5. **Background Sync**: Sync when connectivity restored
6. **Conflict Resolution**: Handle server conflicts gracefully

### Sync Strategy
- **Optimistic UI**: Update UI immediately, sync in background
- **Conflict Resolution**: Last-write-wins with manual merge for conflicts
- **Priority Queue**: High-priority operations (messages) sync first
- **Batch Processing**: Group operations for efficient sync
- **Retry Logic**: Exponential backoff for failed operations

### Storage Strategy
- **AsyncStorage**: Simple key-value data
- **SQLite**: Complex relational data
- **File System**: Large files (documents, media)
- **Memory Cache**: Frequently accessed data

## Testing

### Running Tests
```bash
# Unit tests
npm test

# E2E tests
npm run test:e2e

# Test coverage
npm run test:coverage
```

### Test Structure
```
__tests__/
├── components/         # Component tests
├── services/           # Service tests
├── store/              # Redux tests
├── integration/        # Integration tests
└── e2e/               # End-to-end tests
```

## Building for Production

### iOS Build
```bash
# Development build
eas build --platform ios --profile development

# Production build
eas build --platform ios --profile production

# Submit to App Store
eas submit --platform ios
```

### Android Build
```bash
# Development build
eas build --platform android --profile development

# Production build
eas build --platform android --profile production

# Submit to Google Play
eas submit --platform android
```

### Build Profiles
Configured in `eas.json`:
- **development**: Debug builds with dev settings
- **preview**: Release builds for testing
- **production**: Production builds for app stores

## Performance Optimization

### Bundle Optimization
- Code splitting for large features
- Dynamic imports for screen components
- Image optimization and lazy loading
- Font subsetting for smaller bundles

### Runtime Optimization
- Memoization for expensive computations
- Virtualized lists for large datasets
- Debounced search and input
- Background task optimization

### Storage Optimization
- Data compression for large objects
- Automatic cleanup of old data
- Lazy loading of conversation history
- Efficient caching strategies

## Security

### Authentication
- JWT token-based authentication
- Biometric authentication support
- Secure token storage in Keychain/Keystore
- Automatic token refresh

### Data Protection
- Encrypted local storage for sensitive data
- HTTPS/WSS for all network communication
- Certificate pinning for API calls
- Input validation and sanitization

### Privacy
- Local data encryption
- Secure data deletion
- Privacy-focused analytics
- GDPR compliance features

## Deployment

### Over-the-Air Updates
Using Expo Updates for instant app updates:
```bash
# Publish update
eas update --branch production

# Rollback update
eas update --branch production --message "Rollback"
```

### App Store Deployment
1. **Build**: Create production build with EAS Build
2. **Test**: Test build on physical devices
3. **Submit**: Submit to app stores with EAS Submit
4. **Monitor**: Monitor crashes and performance

## Monitoring and Analytics

### Error Tracking
- Crash reporting with detailed stack traces
- Performance monitoring
- User session recording
- Custom error boundaries

### Analytics
- User behavior tracking
- Feature usage analytics
- Performance metrics
- Offline usage patterns

### Logging
- Structured logging with log levels
- Remote log collection
- Local log storage for debugging
- Log rotation and cleanup

## Troubleshooting

### Common Issues

**Metro bundler issues**:
```bash
npx expo start --clear
```

**iOS build issues**:
```bash
cd ios && pod install && cd ..
```

**Android build issues**:
```bash
cd android && ./gradlew clean && cd ..
```

**Storage issues**:
```bash
# Clear app data
npx expo start --clear

# Reset AsyncStorage
// In app: await AsyncStorage.clear()
```

### Debug Tools
- **Flipper**: React Native debugging
- **Reactotron**: Redux state inspection
- **Expo Dev Tools**: Expo-specific debugging
- **Chrome DevTools**: Network and performance

## Contributing

### Development Workflow
1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Run linting and tests
5. Submit pull request

### Code Standards
- TypeScript strict mode
- ESLint and Prettier formatting
- Jest testing framework
- Conventional commits

### Testing Requirements
- Unit tests for services and utilities
- Component tests for UI components
- Integration tests for user flows
- E2E tests for critical paths

## Roadmap

### Near Term (v1.1)
- Voice input and speech synthesis
- Enhanced offline document processing
- Improved sync conflict resolution
- Better accessibility support

### Medium Term (v1.2)
- Multi-language support
- Advanced agent customization
- Collaborative features
- Enhanced security features

### Long Term (v2.0)
- AR/VR integration
- Advanced AI features
- Enterprise features
- Platform expansion

## License

MIT License - see LICENSE file for details.