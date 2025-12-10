/**
 * Session Sharing - QR code generation and cross-device session continuity
 * Provides sharing functionality with QR codes and real-time sync indicators
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Share2,
  QrCode,
  ExternalLink,
  Copy,
  Check,
  Globe,
  Smartphone,
  Monitor,
  Tablet,
  Users,
  Clock,
  Wifi,
  WifiOff,
  Shield,
  Eye,
  EyeOff,
  Link,
  Settings,
  Download,
  Upload,
  Loader2,
  AlertCircle,
  Info
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { useSessionStore, useActiveSession } from '@/lib/session-store';
import { Session } from '@/lib/opencode-client';

interface SessionSharingProps {
  /**
   * Session to share
   */
  session: Session;
  /**
   * Whether the sharing dialog is open
   */
  open: boolean;
  /**
   * Callback when dialog should close
   */
  onOpenChange: (open: boolean) => void;
}

interface ShareSettings {
  isPublic: boolean;
  allowEditing: boolean;
  expiresAt?: number;
  accessLevel: 'view' | 'comment' | 'edit';
  requireAuth: boolean;
  allowedDomains?: string[];
}

interface ConnectedDevice {
  id: string;
  name: string;
  type: 'desktop' | 'mobile' | 'tablet';
  platform: string;
  lastSeen: number;
  isActive: boolean;
  location?: string;
}

interface ShareAnalytics {
  totalViews: number;
  uniqueVisitors: number;
  lastAccessed: number;
  topCountries: Array<{ country: string; views: number }>;
  deviceTypes: Array<{ type: string; count: number }>;
}

const SHARE_EXPIRY_OPTIONS = [
  { label: 'Never', value: undefined },
  { label: '1 Hour', value: 60 * 60 * 1000 },
  { label: '24 Hours', value: 24 * 60 * 60 * 1000 },
  { label: '7 Days', value: 7 * 24 * 60 * 60 * 1000 },
  { label: '30 Days', value: 30 * 24 * 60 * 60 * 1000 }
];

export const SessionSharing: React.FC<SessionSharingProps> = ({
  session,
  open,
  onOpenChange
}) => {
  const { actions } = useSessionStore();
  
  const [shareUrl, setShareUrl] = useState('');
  const [qrCodeUrl, setQrCodeUrl] = useState('');
  const [shareSettings, setShareSettings] = useState<ShareSettings>({
    isPublic: false,
    allowEditing: false,
    accessLevel: 'view',
    requireAuth: false
  });
  const [connectedDevices, setConnectedDevices] = useState<ConnectedDevice[]>([]);
  const [shareAnalytics, setShareAnalytics] = useState<ShareAnalytics | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isShared, setIsShared] = useState(false);
  const [copied, setCopied] = useState(false);
  const [currentTab, setCurrentTab] = useState('share');

  // Mock data for demonstration
  useEffect(() => {
    if (open && session) {
      // Load existing share settings
      setIsShared(session.shared || false);
      
      if (session.shared) {
        setShareUrl(`https://opencode.dev/s/${session.id}`);
        generateQRCode(`https://opencode.dev/s/${session.id}`);
        loadConnectedDevices();
        loadShareAnalytics();
      }
    }
  }, [open, session]);

  const generateQRCode = (url: string) => {
    // Generate QR code using a service or library
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(url)}&bgcolor=ffffff&color=000000`;
    setQrCodeUrl(qrUrl);
  };

  const loadConnectedDevices = () => {
    // Mock connected devices
    const devices: ConnectedDevice[] = [
      {
        id: 'device-1',
        name: 'MacBook Pro',
        type: 'desktop',
        platform: 'macOS',
        lastSeen: Date.now() - 300000,
        isActive: true,
        location: 'San Francisco, CA'
      },
      {
        id: 'device-2',
        name: 'iPhone',
        type: 'mobile',
        platform: 'iOS',
        lastSeen: Date.now() - 1800000,
        isActive: false,
        location: 'San Francisco, CA'
      }
    ];
    setConnectedDevices(devices);
  };

  const loadShareAnalytics = () => {
    // Mock analytics data
    const analytics: ShareAnalytics = {
      totalViews: 23,
      uniqueVisitors: 8,
      lastAccessed: Date.now() - 600000,
      topCountries: [
        { country: 'United States', views: 15 },
        { country: 'Canada', views: 5 },
        { country: 'United Kingdom', views: 3 }
      ],
      deviceTypes: [
        { type: 'Desktop', count: 12 },
        { type: 'Mobile', count: 8 },
        { type: 'Tablet', count: 3 }
      ]
    };
    setShareAnalytics(analytics);
  };

  const handleCreateShare = async () => {
    setIsGenerating(true);
    try {
      // Create share link with settings
      const url = await actions.shareSession(session.id);
      setShareUrl(url);
      generateQRCode(url);
      setIsShared(true);
      loadConnectedDevices();
      loadShareAnalytics();
    } catch (error) {
      console.error('Failed to create share:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleUpdateSettings = async () => {
    try {
      // Update share settings
      console.log('Updating share settings:', shareSettings);
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  };

  const handleRevokeShare = async () => {
    if (confirm('Are you sure you want to revoke sharing? The link will no longer work.')) {
      try {
        // Revoke share
        setIsShared(false);
        setShareUrl('');
        setQrCodeUrl('');
        setConnectedDevices([]);
        setShareAnalytics(null);
      } catch (error) {
        console.error('Failed to revoke share:', error);
      }
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'desktop':
        return <Monitor className="h-4 w-4" />;
      case 'mobile':
        return <Smartphone className="h-4 w-4" />;
      case 'tablet':
        return <Tablet className="h-4 w-4" />;
      default:
        return <Monitor className="h-4 w-4" />;
    }
  };

  const formatLastSeen = (timestamp: number) => {
    const diffInMinutes = Math.abs(Date.now() - timestamp) / (1000 * 60);
    
    if (diffInMinutes < 1) {
      return 'Just now';
    } else if (diffInMinutes < 60) {
      return `${Math.floor(diffInMinutes)}m ago`;
    } else if (diffInMinutes < 24 * 60) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return `${Math.floor(diffInMinutes / (24 * 60))}d ago`;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Share2 className="h-5 w-5" />
            <span>Share Session</span>
          </DialogTitle>
          <DialogDescription>
            Share &quot;<span className="font-medium">{session.name || session.id}</span>&quot; with others or across your devices
          </DialogDescription>
        </DialogHeader>

        <Tabs value={currentTab} onValueChange={setCurrentTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="share">Share</TabsTrigger>
            <TabsTrigger value="devices">Devices</TabsTrigger>
            <TabsTrigger value="analytics" disabled={!isShared}>Analytics</TabsTrigger>
          </TabsList>

          <div className="mt-6 max-h-[60vh] overflow-auto">
            {/* Share Tab */}
            <TabsContent value="share" className="space-y-6">
              {!isShared ? (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto">
                    <Share2 className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium mb-2">Share Your Session</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Create a shareable link to collaborate with others or continue on different devices
                    </p>
                  </div>
                  
                  {/* Share Settings */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Sharing Options</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label className="text-sm font-medium">Public Access</Label>
                          <p className="text-xs text-muted-foreground">Anyone with the link can access</p>
                        </div>
                        <Switch
                          checked={shareSettings.isPublic}
                          onCheckedChange={(checked) => 
                            setShareSettings({ ...shareSettings, isPublic: checked })
                          }
                        />
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div>
                          <Label className="text-sm font-medium">Allow Editing</Label>
                          <p className="text-xs text-muted-foreground">Others can modify the session</p>
                        </div>
                        <Switch
                          checked={shareSettings.allowEditing}
                          onCheckedChange={(checked) => 
                            setShareSettings({ ...shareSettings, allowEditing: checked })
                          }
                        />
                      </div>
                      
                      <div>
                        <Label className="text-sm font-medium mb-2 block">Access Level</Label>
                        <Select 
                          value={shareSettings.accessLevel} 
                          onValueChange={(value: any) => 
                            setShareSettings({ ...shareSettings, accessLevel: value })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="view">View Only</SelectItem>
                            <SelectItem value="comment">View & Comment</SelectItem>
                            <SelectItem value="edit">Full Edit Access</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Button 
                    onClick={handleCreateShare}
                    disabled={isGenerating}
                    className="w-full"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Creating Share Link...
                      </>
                    ) : (
                      <>
                        <Share2 className="h-4 w-4 mr-2" />
                        Create Share Link
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* QR Code */}
                  <div className="text-center">
                    <div className="inline-block p-4 bg-white rounded-lg border">
                      {qrCodeUrl ? (
                        <img src={qrCodeUrl} alt="Session QR Code" className="w-40 h-40" />
                      ) : (
                        <div className="w-40 h-40 bg-muted rounded flex items-center justify-center">
                          <QrCode className="h-8 w-8 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Scan with any device to continue the session
                    </p>
                  </div>
                  
                  {/* Share URL */}
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Share URL</Label>
                    <div className="flex space-x-2">
                      <Input
                        value={shareUrl}
                        readOnly
                        className="font-mono text-sm"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => copyToClipboard(shareUrl)}
                      >
                        {copied ? (
                          <Check className="h-4 w-4 text-green-500" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => window.open(shareUrl, '_blank')}
                      >
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  
                  {/* Share Status */}
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full" />
                          <div>
                            <p className="text-sm font-medium">Session is shared</p>
                            <p className="text-xs text-muted-foreground">
                              {shareSettings.accessLevel} access • {shareSettings.isPublic ? 'Public' : 'Private'}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleRevokeShare}
                        >
                          Revoke
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                  
                  {/* Cross-device info */}
                  <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                    <div className="flex items-start space-x-3">
                      <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                      <div>
                        <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-1">
                          Cross-device Continuity
                        </h4>
                        <p className="text-sm text-blue-700 dark:text-blue-300">
                          Changes sync in real-time across all connected devices. The session state is automatically saved every few minutes.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* Devices Tab */}
            <TabsContent value="devices" className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">Connected Devices</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Devices that have accessed this session
                </p>
              </div>
              
              {connectedDevices.length === 0 ? (
                <div className="text-center py-8">
                  <Monitor className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground">
                    No devices connected yet
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {connectedDevices.map((device) => (
                    <Card key={device.id}>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <div className={cn(
                              "p-2 rounded-lg",
                              device.isActive ? "bg-green-100 dark:bg-green-900" : "bg-gray-100 dark:bg-gray-800"
                            )}>
                              {getDeviceIcon(device.type)}
                            </div>
                            <div>
                              <div className="flex items-center space-x-2">
                                <p className="font-medium text-sm">{device.name}</p>
                                {device.isActive && (
                                  <div className="flex items-center space-x-1">
                                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                                    <span className="text-xs text-green-600 dark:text-green-400">Active</span>
                                  </div>
                                )}
                              </div>
                              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                                <span>{device.platform}</span>
                                <span>•</span>
                                <span>{formatLastSeen(device.lastSeen)}</span>
                                {device.location && (
                                  <>
                                    <span>•</span>
                                    <span>{device.location}</span>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                          
                          <Badge variant={device.isActive ? "default" : "outline"}>
                            {device.isActive ? (
                              <Wifi className="h-3 w-3 mr-1" />
                            ) : (
                              <WifiOff className="h-3 w-3 mr-1" />
                            )}
                            {device.isActive ? 'Online' : 'Offline'}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics" className="space-y-4">
              {shareAnalytics ? (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-medium mb-2">Share Analytics</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Insights about who has accessed your shared session
                    </p>
                  </div>
                  
                  {/* Overview Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Card>
                      <CardContent className="p-4 text-center">
                        <p className="text-2xl font-bold">{shareAnalytics.totalViews}</p>
                        <p className="text-xs text-muted-foreground">Total Views</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <p className="text-2xl font-bold">{shareAnalytics.uniqueVisitors}</p>
                        <p className="text-xs text-muted-foreground">Unique Visitors</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <p className="text-2xl font-bold">{connectedDevices.length}</p>
                        <p className="text-xs text-muted-foreground">Connected Devices</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <p className="text-2xl font-bold">
                          {formatLastSeen(shareAnalytics.lastAccessed)}
                        </p>
                        <p className="text-xs text-muted-foreground">Last Accessed</p>
                      </CardContent>
                    </Card>
                  </div>
                  
                  {/* Top Countries */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Top Countries</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {shareAnalytics.topCountries.map((country, index) => (
                          <div key={country.country} className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="text-sm font-medium">#{index + 1}</span>
                              <span className="text-sm">{country.country}</span>
                            </div>
                            <Badge variant="outline">{country.views} views</Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                  
                  {/* Device Types */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Device Types</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {shareAnalytics.deviceTypes.map((device) => (
                          <div key={device.type} className="flex items-center justify-between">
                            <span className="text-sm">{device.type}</span>
                            <Badge variant="outline">{device.count}</Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              ) : (
                <div className="text-center py-8">
                  <AlertCircle className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground">
                    Analytics will be available once the session is shared
                  </p>
                </div>
              )}
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};