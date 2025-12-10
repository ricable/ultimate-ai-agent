import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, CheckCircle, AlertCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ToastProps {
  message: string;
  type: "success" | "error" | "info";
  onDismiss: () => void;
  duration?: number;
}

export const Toast: React.FC<ToastProps> = ({
  message,
  type,
  onDismiss,
  duration = 5000,
}) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onDismiss, 200);
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onDismiss]);

  const getIcon = () => {
    switch (type) {
      case "success":
        return <CheckCircle className="h-4 w-4" />;
      case "error":
        return <AlertCircle className="h-4 w-4" />;
      case "info":
        return <Info className="h-4 w-4" />;
    }
  };

  const getColorClasses = () => {
    switch (type) {
      case "success":
        return "bg-success/10 border-success/20 text-success-foreground";
      case "error":
        return "bg-destructive/10 border-destructive/20 text-destructive-foreground";
      case "info":
        return "bg-info/10 border-info/20 text-info-foreground";
    }
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -50, scale: 0.95 }}
          className={cn(
            "flex items-center justify-between p-3 rounded-lg border shadow-md min-w-[300px] max-w-[500px]",
            getColorClasses()
          )}
        >
          <div className="flex items-center space-x-2">
            {getIcon()}
            <span className="text-sm font-medium">{message}</span>
          </div>
          <button
            onClick={() => {
              setIsVisible(false);
              setTimeout(onDismiss, 200);
            }}
            className="ml-4 text-current hover:opacity-70 transition-opacity"
          >
            <X className="h-4 w-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export interface ToastContainerProps {
  children: React.ReactNode;
  className?: string;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({
  children,
  className,
}) => {
  return (
    <div
      className={cn(
        "fixed top-4 right-4 z-50 flex flex-col space-y-2",
        className
      )}
    >
      {children}
    </div>
  );
};