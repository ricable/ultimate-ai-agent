"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { FileIcon } from "lucide-react";

interface FileEntry {
  path: string;
  name: string;
  type: "file" | "directory";
  is_directory: boolean;
  size: number;
}

interface FilePickerProps {
  basePath: string;
  onSelect: (entry: FileEntry) => void;
  onClose: () => void;
  initialQuery: string;
  className?: string;
}

export const FilePicker: React.FC<FilePickerProps> = ({
  basePath,
  onSelect,
  onClose,
  initialQuery,
  className = ""
}) => {
  const handleFileSelect = () => {
    // Placeholder implementation - in real app would show file browser
    onSelect({
      path: `${basePath}/example.txt`,
      name: "example.txt",
      type: "file",
      is_directory: false,
      size: 1024
    });
    onClose();
  };

  return (
    <div className={`border rounded-lg p-4 bg-background ${className}`}>
      <div className="mb-2">
        <p className="text-sm text-muted-foreground">Select a file from {basePath}</p>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={handleFileSelect}
      >
        <FileIcon className="h-4 w-4 mr-2" />
        Select Example File
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={onClose}
        className="ml-2"
      >
        Cancel
      </Button>
    </div>
  );
};