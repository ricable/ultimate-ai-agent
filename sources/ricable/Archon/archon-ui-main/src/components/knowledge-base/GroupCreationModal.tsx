import { useState } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { KnowledgeItem, knowledgeBaseService } from '../../services/knowledgeBaseService';
import { useToast } from '../../contexts/ToastContext';

interface GroupCreationModalProps {
  selectedItems: KnowledgeItem[];
  onClose: () => void;
  onSuccess: () => void;
}

export const GroupCreationModal = ({ selectedItems, onClose, onSuccess }: GroupCreationModalProps) => {
  const [groupName, setGroupName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { showToast } = useToast();

  const handleCreateGroup = async () => {
    if (!groupName.trim()) {
      showToast('Please enter a group name', 'error');
      return;
    }

    setIsLoading(true);
    try {
      // Update each selected item with the group name
      const updatePromises = selectedItems.map(item =>
        knowledgeBaseService.updateKnowledgeItem(item.source_id, {
          ...item.metadata,
          group_name: groupName.trim()
        })
      );

      await Promise.all(updatePromises);
      
      showToast(`Successfully created group "${groupName}" with ${selectedItems.length} items`, 'success');
      onSuccess();
    } catch (error) {
      console.error('Error creating group:', error);
      showToast('Failed to create group', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: "spring", duration: 0.3 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-2xl"
        >
          <Card className="relative">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
                Create Knowledge Group
              </h2>
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            {/* Group Name Input */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-zinc-300 mb-2">
                Group Name
              </label>
              <Input
                value={groupName}
                onChange={(e) => setGroupName(e.target.value)}
                placeholder="Enter group name..."
                className="w-full"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isLoading) {
                    handleCreateGroup();
                  }
                }}
              />
            </div>

            {/* Selected Items Preview */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 dark:text-zinc-300 mb-3">
                Items to be grouped ({selectedItems.length})
              </h3>
              <div className="max-h-60 overflow-y-auto space-y-2 pr-2">
                {selectedItems.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 bg-gray-50 dark:bg-zinc-800/50 rounded-lg"
                  >
                    <h4 className="font-medium text-gray-800 dark:text-white text-sm">
                      {item.title}
                    </h4>
                    <p className="text-xs text-gray-600 dark:text-zinc-400 mt-1 line-clamp-1">
                      {item.metadata.description || item.source_id}
                    </p>
                    {item.metadata.tags && item.metadata.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {item.metadata.tags.slice(0, 3).map((tag, index) => (
                          <Badge key={index} accentColor="gray">
                            {tag}
                          </Badge>
                        ))}
                        {item.metadata.tags.length > 3 && (
                          <Badge accentColor="gray">
                            +{item.metadata.tags.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3">
              <Button
                variant="ghost"
                onClick={onClose}
                disabled={isLoading}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                accentColor="blue"
                onClick={handleCreateGroup}
                disabled={isLoading || !groupName.trim()}
              >
                {isLoading ? 'Creating...' : 'Create Group'}
              </Button>
            </div>
          </Card>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};