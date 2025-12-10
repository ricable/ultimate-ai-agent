# Socket & Memoization Patterns

## Quick Reference

### DO:
- ✅ Track optimistic updates to prevent double-renders
- ✅ Memoize socket event handlers with useCallback
- ✅ Check if incoming data actually differs from current state
- ✅ Use debouncing for rapid UI updates (drag & drop)
- ✅ Clean up socket listeners in useEffect cleanup

### DON'T:
- ❌ Update state without checking if data changed
- ❌ Create new handler functions on every render
- ❌ Apply server updates that match pending optimistic updates
- ❌ Forget to handle the "modal open" edge case

## Pattern Examples

### Optimistic Update Pattern

```typescript
import { useOptimisticUpdates } from '../../hooks/useOptimisticUpdates';

const MyComponent = () => {
  const { addPendingUpdate, isPendingUpdate } = useOptimisticUpdates<Task>();
  
  const handleLocalUpdate = (task: Task) => {
    // Track the optimistic update
    addPendingUpdate({
      id: task.id,
      timestamp: Date.now(),
      data: task,
      operation: 'update'
    });
    
    // Update local state immediately
    setTasks(prev => prev.map(t => t.id === task.id ? task : t));
    
    // Persist to server
    api.updateTask(task);
  };
  
  const handleServerUpdate = useCallback((task: Task) => {
    // Skip if this is our own update echoing back
    if (isPendingUpdate(task.id, task)) {
      console.log('Skipping own optimistic update');
      return;
    }
    
    // Apply server update
    setTasks(prev => prev.map(t => t.id === task.id ? task : t));
  }, [isPendingUpdate]);
};
```

### Socket Handler Pattern

```typescript
import { useSocketSubscription } from '../../hooks/useSocketSubscription';

const MyComponent = () => {
  // Option 1: Using the hook
  useSocketSubscription(
    socketService,
    'data_updated',
    (data) => {
      console.log('Data updated:', data);
      // Handle update
    },
    [/* dependencies */]
  );
  
  // Option 2: Manual memoization
  const handleUpdate = useCallback((message: any) => {
    const data = message.data || message;
    
    setItems(prev => {
      // Check if data actually changed
      const existing = prev.find(item => item.id === data.id);
      if (existing && JSON.stringify(existing) === JSON.stringify(data)) {
        return prev; // No change, prevent re-render
      }
      
      return prev.map(item => item.id === data.id ? data : item);
    });
  }, []);
  
  useEffect(() => {
    socketService.addMessageHandler('update', handleUpdate);
    return () => {
      socketService.removeMessageHandler('update', handleUpdate);
    };
  }, [handleUpdate]);
};
```

### Debounced Reordering Pattern

```typescript
const useReordering = () => {
  const debouncedPersist = useMemo(
    () => debounce(async (items: Item[]) => {
      try {
        await api.updateOrder(items);
      } catch (error) {
        console.error('Failed to persist order:', error);
        // Rollback or retry logic
      }
    }, 500),
    []
  );
  
  const handleReorder = useCallback((dragIndex: number, dropIndex: number) => {
    // Update UI immediately
    setItems(prev => {
      const newItems = [...prev];
      const [draggedItem] = newItems.splice(dragIndex, 1);
      newItems.splice(dropIndex, 0, draggedItem);
      
      // Update order numbers
      return newItems.map((item, index) => ({
        ...item,
        order: index + 1
      }));
    });
    
    // Persist changes (debounced)
    debouncedPersist(items);
  }, [items, debouncedPersist]);
};
```

## WebSocket Service Configuration

### Deduplication

The enhanced WebSocketService now includes automatic deduplication:

```typescript
// Configure deduplication window (default: 100ms)
socketService.setDeduplicationWindow(200); // 200ms window

// Duplicate messages within the window are automatically filtered
```

### Connection Management

```typescript
// Always check connection state before critical operations
if (socketService.isConnected()) {
  socketService.send({ type: 'update', data: payload });
}

// Monitor connection state
socketService.addStateChangeHandler((state) => {
  if (state === WebSocketState.CONNECTED) {
    console.log('Connected - refresh data');
  }
});
```

## Common Patterns

### 1. State Equality Checks

Always check if incoming data actually differs from current state:

```typescript
// ❌ BAD - Always triggers re-render
setTasks(prev => prev.map(t => t.id === id ? newTask : t));

// ✅ GOOD - Only updates if changed
setTasks(prev => {
  const existing = prev.find(t => t.id === id);
  if (existing && deepEqual(existing, newTask)) return prev;
  return prev.map(t => t.id === id ? newTask : t);
});
```

### 2. Modal State Handling

Be aware of modal state when applying updates:

```typescript
const handleSocketUpdate = useCallback((data) => {
  if (isModalOpen && editingItem?.id === data.id) {
    console.warn('Update received while editing - consider skipping or merging');
    // Option 1: Skip the update
    // Option 2: Merge with current edits
    // Option 3: Show conflict resolution UI
  }
  
  // Normal update flow
}, [isModalOpen, editingItem]);
```

### 3. Cleanup Pattern

Always clean up socket listeners:

```typescript
useEffect(() => {
  const handlers = [
    { event: 'create', handler: handleCreate },
    { event: 'update', handler: handleUpdate },
    { event: 'delete', handler: handleDelete }
  ];
  
  // Add all handlers
  handlers.forEach(({ event, handler }) => {
    socket.addMessageHandler(event, handler);
  });
  
  // Cleanup
  return () => {
    handlers.forEach(({ event, handler }) => {
      socket.removeMessageHandler(event, handler);
    });
  };
}, [handleCreate, handleUpdate, handleDelete]);
```

## Performance Tips

1. **Measure First**: Use React DevTools Profiler before optimizing
2. **Batch Updates**: Group related state changes
3. **Debounce Rapid Changes**: Especially for drag & drop operations
4. **Use Stable References**: Memoize callbacks passed to child components
5. **Avoid Deep Equality Checks**: Use optimized comparison for large objects

## Debugging

Enable verbose logging for troubleshooting:

```typescript
// In development
if (process.env.NODE_ENV === 'development') {
  console.log('[Socket] Message received:', message);
  console.log('[Socket] Deduplication result:', isDuplicate);
  console.log('[Optimistic] Pending updates:', pendingUpdates);
}
```

## Migration Guide

To migrate existing components:

1. Import `useOptimisticUpdates` hook
2. Wrap socket handlers with `useCallback`
3. Add optimistic update tracking to local changes
4. Check for pending updates in socket handlers
5. Test with React DevTools Profiler

Remember: The goal is to eliminate unnecessary re-renders while maintaining real-time synchronization across all connected clients.