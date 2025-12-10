// File: frontend/src/App.tsx
import './App.css'
import { MainDashboard } from './components/dashboard/MainDashboard'
import { AuthProvider, AuthPage, ProtectedRoute } from './auth'
import { useAuth } from './auth/AuthContext'

function AppContent() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="w-full h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthPage />;
  }

  return (
    <ProtectedRoute>
      <MainDashboard />
    </ProtectedRoute>
  );
}

function App() {
  return (
    <AuthProvider>
      <div className="App w-full h-full">
        <AppContent />
      </div>
    </AuthProvider>
  )
}

export default App