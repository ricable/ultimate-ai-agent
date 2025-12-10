import { configureStore, combineReducers } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { agentsReducer } from './slices/agentsSlice';
import { chatReducer } from './slices/chatSlice';
import { documentsReducer } from './slices/documentsSlice';
import { syncReducer } from './slices/syncSlice';
import { authReducer } from './slices/authSlice';
import { offlineReducer } from './slices/offlineSlice';

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['agents', 'chat', 'documents', 'auth', 'offline'],
  blacklist: ['sync'], // Don't persist sync state
};

const rootReducer = combineReducers({
  agents: agentsReducer,
  chat: chatReducer,
  documents: documentsReducer,
  sync: syncReducer,
  auth: authReducer,
  offline: offlineReducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

export const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
});

export const persistor = persistStore(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;