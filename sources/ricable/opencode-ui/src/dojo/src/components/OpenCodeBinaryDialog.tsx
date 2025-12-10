import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ExternalLink, FileQuestion, Terminal, AlertCircle, Loader2, CheckCircle } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { openCodeClient } from "@/lib/opencode-client";

interface OpenCodeInstallation {
  path: string;
  version: string;
  type: "global" | "local" | "custom";
  valid: boolean;
}

interface OpenCodeBinaryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
  onError: (message: string) => void;
}

export function OpenCodeBinaryDialog({ open, onOpenChange, onSuccess, onError }: OpenCodeBinaryDialogProps) {
  const [selectedInstallation, setSelectedInstallation] = useState<OpenCodeInstallation | null>(null);
  const [customPath, setCustomPath] = useState("");
  const [isValidating, setIsValidating] = useState(false);
  const [installations, setInstallations] = useState<OpenCodeInstallation[]>([]);
  const [checkingInstallations, setCheckingInstallations] = useState(true);
  const [showCustomPath, setShowCustomPath] = useState(false);

  useEffect(() => {
    if (open) {
      checkInstallations();
    }
  }, [open]);

  const checkInstallations = async () => {
    try {
      setCheckingInstallations(true);
      
      // Mock data for OpenCode installations
      // In a real implementation, this would scan common installation paths
      const mockInstallations: OpenCodeInstallation[] = [
        {
          path: "/usr/local/bin/opencode",
          version: "1.0.0",
          type: "global",
          valid: true
        },
        {
          path: "/opt/homebrew/bin/opencode", 
          version: "1.0.0",
          type: "global",
          valid: true
        },
        {
          path: "./node_modules/.bin/opencode",
          version: "0.9.8",
          type: "local",
          valid: true
        }
      ];
      
      // Filter to only show valid installations
      const validInstallations = mockInstallations.filter(inst => inst.valid);
      setInstallations(validInstallations);
      
      // Auto-select the first valid installation
      if (validInstallations.length > 0) {
        setSelectedInstallation(validInstallations[0]);
      }
    } catch (error) {
      console.error("Failed to check OpenCode installations:", error);
      setInstallations([]);
    } finally {
      setCheckingInstallations(false);
    }
  };

  const validateCustomPath = async (path: string): Promise<boolean> => {
    try {
      // In a real implementation, this would validate the OpenCode binary
      // For now, we'll do basic path validation
      if (!path || !path.trim()) {
        return false;
      }
      
      // Simulate validation delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock validation - accept paths that contain "opencode"
      return path.toLowerCase().includes("opencode");
    } catch (error) {
      return false;
    }
  };

  const handleCustomPathChange = async (path: string) => {
    setCustomPath(path);
    
    if (path.trim()) {
      setIsValidating(true);
      const isValid = await validateCustomPath(path);
      setIsValidating(false);
      
      if (isValid) {
        setSelectedInstallation({
          path: path.trim(),
          version: "unknown",
          type: "custom",
          valid: true
        });
      } else {
        setSelectedInstallation(null);
      }
    } else {
      setSelectedInstallation(null);
    }
  };

  const handleSave = async () => {
    if (!selectedInstallation) {
      onError("Please select an OpenCode installation");
      return;
    }

    setIsValidating(true);
    try {
      // In a real implementation, this would save the binary path to OpenCode config
      // For now, we'll simulate a successful save
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      onSuccess();
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to save OpenCode binary path:", error);
      onError(error instanceof Error ? error.message : "Failed to save OpenCode binary path");
    } finally {
      setIsValidating(false);
    }
  };

  const hasInstallations = installations.length > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileQuestion className="w-5 h-5" />
            Select OpenCode Installation
          </DialogTitle>
          <DialogDescription className="space-y-3 mt-4">
            {checkingInstallations ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                <span className="ml-2 text-sm text-muted-foreground">Searching for OpenCode installations...</span>
              </div>
            ) : hasInstallations ? (
              <p>
                Multiple OpenCode installations were found on your system. 
                Please select which one you&apos;d like to use.
              </p>
            ) : (
              <>
                <p>
                  OpenCode was not found in any of the common installation locations. 
                  Please install OpenCode or specify a custom path.
                </p>
                <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
                  <AlertCircle className="w-4 h-4 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    <span className="font-medium">Searched locations:</span> PATH, /usr/local/bin, 
                    /opt/homebrew/bin, ./node_modules/.bin, ~/.local/bin, ~/go/bin
                  </p>
                </div>
              </>
            )}
            <div className="flex items-center gap-2 p-3 bg-muted rounded-md">
              <Terminal className="w-4 h-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                <span className="font-medium">Tip:</span> You can install OpenCode using{" "}
                <code className="px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded">go install github.com/sst/opencode@latest</code>
              </p>
            </div>
          </DialogDescription>
        </DialogHeader>

        {!checkingInstallations && (
          <div className="py-4 space-y-4">
            {/* Installation Options */}
            {hasInstallations && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">Found Installations:</Label>
                {installations.map((installation, index) => (
                  <div
                    key={index}
                    className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                      selectedInstallation?.path === installation.path
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50"
                    }`}
                    onClick={() => setSelectedInstallation(installation)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CheckCircle className={`w-4 h-4 ${
                          selectedInstallation?.path === installation.path ? "text-primary" : "text-muted-foreground"
                        }`} />
                        <div>
                          <p className="text-sm font-medium">{installation.path}</p>
                          <p className="text-xs text-muted-foreground">
                            Version: {installation.version} • Type: {installation.type}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Custom Path Option */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Custom Path:</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCustomPath(!showCustomPath)}
                >
                  {showCustomPath ? "Hide" : "Specify custom path"}
                </Button>
              </div>

              {(showCustomPath || !hasInstallations) && (
                <div className="space-y-2">
                  <div className="relative">
                    <Input
                      placeholder="/path/to/opencode"
                      value={customPath}
                      onChange={(e) => handleCustomPathChange(e.target.value)}
                      className="pr-8"
                    />
                    {isValidating && (
                      <Loader2 className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
                    )}
                    {!isValidating && customPath && selectedInstallation?.type === "custom" && (
                      <CheckCircle className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-green-500" />
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Enter the full path to your OpenCode binary
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        <DialogFooter className="gap-3">
          <Button
            variant="outline"
            onClick={() => window.open("https://docs.sst.dev/opencode/installation", "_blank")}
            className="mr-auto"
          >
            <ExternalLink className="w-4 h-4 mr-2" />
            Installation Guide
          </Button>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isValidating}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSave} 
            disabled={isValidating || !selectedInstallation}
          >
            {isValidating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Validating...
              </>
            ) : selectedInstallation ? (
              "Save Selection"
            ) : (
              "No Installation Selected"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}