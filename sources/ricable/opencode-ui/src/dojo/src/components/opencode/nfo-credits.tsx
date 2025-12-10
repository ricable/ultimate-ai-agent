import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Volume2, VolumeX, Github, ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface NFOCreditsProps {
  /**
   * Callback when the NFO window is closed
   */
  onClose: () => void;
}

/**
 * NFO Credits component - Displays a keygen/crack style credits window
 * Adapted for OpenCode with multi-provider ecosystem credits
 * 
 * @example
 * <NFOCredits onClose={() => setShowNFO(false)} />
 */
export const NFOCredits: React.FC<NFOCreditsProps> = ({ onClose }) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollPosition, setScrollPosition] = useState(0);
  
  // Start auto-scrolling
  useEffect(() => {
    const scrollInterval = setInterval(() => {
      setScrollPosition(prev => prev + 1);
    }, 30); // Smooth scrolling speed
    
    return () => clearInterval(scrollInterval);
  }, []);
  
  // Apply scroll position
  useEffect(() => {
    if (scrollRef.current) {
      const maxScroll = scrollRef.current.scrollHeight - scrollRef.current.clientHeight;
      if (scrollPosition >= maxScroll) {
        // Reset to beginning when reaching the end
        setScrollPosition(0);
        scrollRef.current.scrollTop = 0;
      } else {
        scrollRef.current.scrollTop = scrollPosition;
      }
    }
  }, [scrollPosition]);

  const handleOpenUrl = (url: string) => {
    window.open(url, '_blank');
  };
  
  // Credits content adapted for OpenCode
  const creditsContent = [
    { type: "header", text: "OPENCODE DOJO v1.0.0" },
    { type: "subheader", text: "[ MULTI-PROVIDER AI CODING REVOLUTION ]" },
    { type: "spacer" },
    { type: "section", title: "━━━ CORE PROVIDERS ━━━" },
    { type: "credit", role: "ANTHROPIC", name: "Claude 3.5 Sonnet & Haiku" },
    { type: "credit", role: "OPENAI", name: "GPT-4o & O1 Series" },
    { type: "credit", role: "GROQ", name: "Llama 3.1 & Mixtral" },
    { type: "credit", role: "GOOGLE", name: "Gemini 2.0 Flash" },
    { type: "spacer" },
    { type: "section", title: "━━━ LOCAL PROVIDERS ━━━" },
    { type: "credit", role: "OLLAMA", name: "Local Model Runtime" },
    { type: "credit", role: "LLAMA.CPP", name: "High-Performance Inference" },
    { type: "credit", role: "LM STUDIO", name: "Local Model Management" },
    { type: "credit", role: "LOCALAI", name: "OpenAI-Compatible API" },
    { type: "spacer" },
    { type: "section", title: "━━━ FRAMEWORK STACK ━━━" },
    { type: "credit", role: "RUNTIME", name: "Next.js 15 + React 18" },
    { type: "credit", role: "LANGUAGE", name: "TypeScript" },
    { type: "credit", role: "STYLING", name: "Tailwind CSS + shadcn/ui" },
    { type: "credit", role: "ANIMATIONS", name: "Framer Motion" },
    { type: "credit", role: "PACKAGE MANAGER", name: "pnpm" },
    { type: "credit", role: "TESTING", name: "Vitest + Playwright" },
    { type: "spacer" },
    { type: "section", title: "━━━ CORE FEATURES ━━━" },
    { type: "text", content: "75+ AI Providers with unified interface" },
    { type: "text", content: "Advanced checkpoint system with branching" },
    { type: "text", content: "MCP server integration and marketplace" },
    { type: "text", content: "Real-time usage analytics and cost tracking" },
    { type: "text", content: "Local model support with privacy focus" },
    { type: "spacer" },
    { type: "ascii", content: `
     ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄  ▄▄▄▄▄▄▄ 
    █       █       █       █       █       █       █      ██       █
    █   ▄   █    ▄  █    ▄▄▄█  ▄    █   ▄   █    ▄  █  ▄    █    ▄▄▄█
    █  █ █  █   █▄█ █   █▄▄▄█ █ █   █  █ █  █   █▄█ █ █ █   █   █▄▄▄ 
    █  █▄█  █    ▄▄▄█    ▄▄▄█ █▄█   █  █▄█  █    ▄▄▄█ █▄█   █    ▄▄▄█
    █       █   █   █   █▄▄▄█       █       █   █   █       █   █▄▄▄ 
    █▄▄▄▄▄▄▄█▄▄▄█   █▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄█   █▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█
    ` },
    { type: "spacer" },
    { type: "text", content: "The future of AI-powered development is here!" },
    { type: "text", content: "Multiple providers, unlimited possibilities" },
    { type: "text", content: "Local models, global intelligence" },
    { type: "spacer" },
    { type: "spacer" },
    { type: "spacer" },
  ];
  
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center"
      >
        {/* Backdrop with blur */}
        <div 
          className="absolute inset-0 bg-black/80 backdrop-blur-md"
          onClick={onClose}
        />
        
        {/* NFO Window */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.8, opacity: 0 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative z-10"
        >
          <Card className="w-[600px] h-[500px] bg-background border-border shadow-2xl overflow-hidden">
            {/* Window Header */}
            <div className="flex items-center justify-between px-4 py-2 bg-card border-b border-border">
              <div className="flex items-center space-x-2">
                <div className="text-sm font-bold tracking-wider font-mono text-foreground">
                  OPENCODE.NFO
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleOpenUrl("https://github.com/opencode-ui/opencode/issues/new");
                  }}
                  className="flex items-center gap-1 h-auto px-2 py-1"
                  title="Report an issue"
                >
                  <Github className="h-3 w-3" />
                  <span className="text-xs">Report Issue</span>
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleOpenUrl("https://opencode.dev");
                  }}
                  className="flex items-center gap-1 h-auto px-2 py-1"
                  title="Visit website"
                >
                  <ExternalLink className="h-3 w-3" />
                  <span className="text-xs">Website</span>
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClose();
                  }}
                  className="h-6 w-6 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            {/* NFO Content */}
            <div className="relative h-[calc(100%-40px)] bg-background overflow-hidden">
              {/* OpenCode Logo Section (Fixed at top) */}
              <div className="absolute top-0 left-0 right-0 bg-background z-10 pb-4 text-center">
                <button
                  className="inline-block mt-4 hover:scale-110 transition-transform cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleOpenUrl("https://opencode.dev");
                  }}
                >
                  <div className="h-20 w-20 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-xl">OC</span>
                  </div>
                </button>
                <div className="text-muted-foreground text-sm font-mono mt-2 tracking-wider">
                  Multi-Provider AI Development Platform
                </div>
              </div>
              
              {/* Scrolling Credits */}
              <div 
                ref={scrollRef}
                className="absolute inset-0 top-32 overflow-hidden"
                style={{ fontFamily: "'Courier New', monospace" }}
              >
                <div className="px-8 pb-32">
                  {creditsContent.map((item, index) => {
                    switch (item.type) {
                      case "header":
                        return (
                          <div 
                            key={index} 
                            className="text-foreground text-3xl font-bold text-center mb-2 tracking-widest"
                          >
                            {item.text}
                          </div>
                        );
                      case "subheader":
                        return (
                          <div 
                            key={index} 
                            className="text-muted-foreground text-lg text-center mb-8 tracking-wide"
                          >
                            {item.text}
                          </div>
                        );
                      case "section":
                        return (
                          <div 
                            key={index} 
                            className="text-foreground text-xl font-bold text-center my-6 tracking-wider"
                          >
                            {item.title}
                          </div>
                        );
                      case "credit":
                        return (
                          <div 
                            key={index} 
                            className="flex justify-between items-center mb-2 text-foreground"
                          >
                            <span className="text-sm text-muted-foreground">{item.role}:</span>
                            <span className="text-base tracking-wide">{item.name}</span>
                          </div>
                        );
                      case "text":
                        return (
                          <div 
                            key={index} 
                            className="text-muted-foreground text-center text-sm mb-2"
                          >
                            {item.content}
                          </div>
                        );
                      case "ascii":
                        return (
                          <pre 
                            key={index} 
                            className="text-foreground text-xs text-center my-6 leading-tight opacity-80"
                          >
                            {item.content}
                          </pre>
                        );
                      case "spacer":
                        return <div key={index} className="h-8" />;
                      default:
                        return null;
                    }
                  })}
                </div>
              </div>
              
              {/* Subtle Scanlines Effect */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-foreground/[0.02] to-transparent opacity-50" />
              </div>
            </div>
          </Card>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};