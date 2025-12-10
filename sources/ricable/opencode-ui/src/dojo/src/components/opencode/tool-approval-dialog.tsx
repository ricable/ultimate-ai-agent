/**
 * Tool Approval Dialog - Enhanced security workflow for tool execution approval
 */

"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Shield,
  Terminal,
  Clock,
  User,
  Settings,
  Eye,
  EyeOff
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';

interface ToolApprovalRequest {
  id: string;
  toolId: string;
  toolName: string;
  description: string;
  params: Record<string, any>;
  sessionId?: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  securityWarnings: string[];
  requestedAt: number;
  estimatedDuration?: number;
}

interface ToolApprovalDialogProps {
  request: ToolApprovalRequest | null;
  open: boolean;
  onApprove: (id: string, conditions?: { timeout?: number; note?: string }) => void;
  onDeny: (id: string, reason: string) => void;
  onClose: () => void;
}

export const ToolApprovalDialog: React.FC<ToolApprovalDialogProps> = ({
  request,
  open,
  onApprove,
  onDeny,
  onClose
}) => {
  const [showFullParams, setShowFullParams] = useState(false);
  const [approvalNote, setApprovalNote] = useState('');
  const [customTimeout, setCustomTimeout] = useState(false);
  const [timeoutMinutes, setTimeoutMinutes] = useState(5);
  const [rememberChoice, setRememberChoice] = useState(false);

  if (!request) return null;

  const getRiskConfig = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return {
          color: 'text-red-500',
          bgColor: 'bg-red-500/10',
          borderColor: 'border-red-500/20',
          icon: AlertTriangle,
          label: 'Critical Risk'
        };
      case 'high':
        return {
          color: 'text-orange-500',
          bgColor: 'bg-orange-500/10',
          borderColor: 'border-orange-500/20',
          icon: AlertTriangle,
          label: 'High Risk'
        };
      case 'medium':
        return {
          color: 'text-yellow-500',
          bgColor: 'bg-yellow-500/10',
          borderColor: 'border-yellow-500/20',
          icon: Shield,
          label: 'Medium Risk'
        };
      case 'low':
        return {
          color: 'text-green-500',
          bgColor: 'bg-green-500/10',
          borderColor: 'border-green-500/20',
          icon: CheckCircle,
          label: 'Low Risk'
        };
      default:
        return {
          color: 'text-gray-500',
          bgColor: 'bg-gray-500/10',
          borderColor: 'border-gray-500/20',
          icon: Shield,
          label: 'Unknown Risk'
        };
    }
  };

  const riskConfig = getRiskConfig(request.riskLevel);
  const RiskIcon = riskConfig.icon;

  const handleApprove = () => {
    const conditions: { timeout?: number; note?: string } = {};
    
    if (customTimeout) {
      conditions.timeout = timeoutMinutes * 60 * 1000; // Convert to milliseconds
    }
    
    if (approvalNote.trim()) {
      conditions.note = approvalNote.trim();
    }
    
    onApprove(request.id, conditions);
    resetForm();
  };

  const handleDeny = () => {
    const reason = approvalNote.trim() || 'Denied by user';
    onDeny(request.id, reason);
    resetForm();
  };

  const resetForm = () => {
    setApprovalNote('');
    setCustomTimeout(false);
    setTimeoutMinutes(5);
    setRememberChoice(false);
    setShowFullParams(false);
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatParams = (params: Record<string, any>) => {
    try {
      return JSON.stringify(params, null, 2);
    } catch {
      return String(params);
    }
  };

  const truncateParams = (params: Record<string, any>, maxLength: number = 200) => {
    const formatted = formatParams(params);
    if (formatted.length <= maxLength) return formatted;
    return formatted.substring(0, maxLength) + '...';
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Terminal className="h-5 w-5" />
            <span>Tool Execution Approval Required</span>
          </DialogTitle>
          <DialogDescription>
            Review the tool execution request and security assessment below.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Risk Assessment */}
          <Card className={cn("border-2", riskConfig.borderColor)}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <RiskIcon className={cn("h-5 w-5", riskConfig.color)} />
                  <CardTitle className={cn("text-lg", riskConfig.color)}>
                    {riskConfig.label}
                  </CardTitle>
                </div>
                <Badge variant="outline" className={riskConfig.color}>
                  {request.riskLevel.toUpperCase()}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className={cn("p-4", riskConfig.bgColor)}>
              {request.securityWarnings.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Security Warnings:</p>
                  <ul className="space-y-1">
                    {request.securityWarnings.map((warning, index) => (
                      <li key={index} className="text-sm flex items-start space-x-2">
                        <AlertTriangle className="h-3 w-3 mt-0.5 text-orange-500 flex-shrink-0" />
                        <span>{warning}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-sm">No specific security warnings identified.</p>
              )}
            </CardContent>
          </Card>

          {/* Tool Information */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tool Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Tool Name
                  </p>
                  <p className="text-sm font-medium">{request.toolName}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Tool ID
                  </p>
                  <p className="text-sm font-mono">{request.toolId}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Requested At
                  </p>
                  <p className="text-sm">{formatTimestamp(request.requestedAt)}</p>
                </div>
                {request.estimatedDuration && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      Estimated Duration
                    </p>
                    <p className="text-sm">{request.estimatedDuration}ms</p>
                  </div>
                )}
              </div>
              
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                  Description
                </p>
                <p className="text-sm">{request.description}</p>
              </div>
            </CardContent>
          </Card>

          {/* Parameters */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Execution Parameters</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowFullParams(!showFullParams)}
                  className="h-8"
                >
                  {showFullParams ? (
                    <>
                      <EyeOff className="h-3 w-3 mr-1" />
                      Hide
                    </>
                  ) : (
                    <>
                      <Eye className="h-3 w-3 mr-1" />
                      Show All
                    </>
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <pre className="text-xs bg-muted p-3 rounded overflow-x-auto whitespace-pre-wrap">
                {showFullParams 
                  ? formatParams(request.params)
                  : truncateParams(request.params)
                }
              </pre>
            </CardContent>
          </Card>

          {/* Approval Options */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Approval Options</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Custom Timeout</p>
                  <p className="text-xs text-muted-foreground">
                    Set a specific timeout for this execution
                  </p>
                </div>
                <Switch
                  checked={customTimeout}
                  onCheckedChange={setCustomTimeout}
                />
              </div>
              
              {customTimeout && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2"
                >
                  <label className="text-sm font-medium">Timeout (minutes)</label>
                  <input
                    type="number"
                    min="1"
                    max="60"
                    value={timeoutMinutes}
                    onChange={(e) => setTimeoutMinutes(Number(e.target.value))}
                    className="w-full px-3 py-2 border rounded text-sm"
                  />
                </motion.div>
              )}
              
              <div>
                <label className="text-sm font-medium">Notes (Optional)</label>
                <Textarea
                  placeholder="Add any notes about this approval decision..."
                  value={approvalNote}
                  onChange={(e) => setApprovalNote(e.target.value)}
                  className="mt-1"
                  rows={3}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Remember Choice</p>
                  <p className="text-xs text-muted-foreground">
                    Apply this decision to similar requests
                  </p>
                </div>
                <Switch
                  checked={rememberChoice}
                  onCheckedChange={setRememberChoice}
                />
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-4 border-t">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeny}>
              <XCircle className="h-4 w-4 mr-2" />
              Deny
            </Button>
            <Button onClick={handleApprove}>
              <CheckCircle className="h-4 w-4 mr-2" />
              Approve
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};