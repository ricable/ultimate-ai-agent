import { useEffect, useRef, useState } from 'react';

interface NeonGlowOptions {
  opacity?: number;
  blur?: number;
  size?: number;
  color?: string;
  speed?: number;
  enabled?: boolean;
}

interface NeonGlowHook {
  containerRef: React.RefObject<HTMLDivElement>;
  isAnimating: boolean;
  start: () => void;
  stop: () => void;
  updateOptions: (options: Partial<NeonGlowOptions>) => void;
}

export const useNeonGlow = (initialOptions: NeonGlowOptions = {}): NeonGlowHook => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [options, setOptions] = useState<Required<NeonGlowOptions>>({
    opacity: 0.8,
    blur: 2,
    size: 100,
    color: 'blue',
    speed: 2000,
    enabled: true,
    ...initialOptions
  });

  const animationRef = useRef<number>();
  const elementsRef = useRef<HTMLDivElement[]>([]);

  // Create optimized heart chakra pattern
  const createHeartChakra = () => {
    if (!containerRef.current) return;

    // Clear existing elements
    elementsRef.current.forEach(el => {
      if (containerRef.current?.contains(el)) {
        containerRef.current.removeChild(el);
      }
    });
    elementsRef.current = [];

    const container = containerRef.current;
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    const radius = options.size;

    // Create heart shape using mathematical equation
    // Using fewer points for better performance (20 instead of 100)
    const heartPoints = [];
    for (let i = 0; i < 20; i++) {
      const t = (i / 20) * Math.PI * 2;
      
      // Heart equation: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
      const heartX = centerX + Math.pow(Math.sin(t), 3) * radius * 0.8;
      const heartY = centerY - (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) * radius * 0.04;
      
      heartPoints.push({ x: heartX, y: heartY });
    }

    // Create 12 radiating lines from center (reduced from more for performance)
    const rayPoints = [];
    for (let ray = 0; ray < 12; ray++) {
      const rayAngle = (ray * Math.PI * 2 / 12);
      const rayRadius = radius * 0.8;
      rayPoints.push({
        x: centerX + Math.cos(rayAngle) * rayRadius,
        y: centerY + Math.sin(rayAngle) * rayRadius
      });
    }

    // Create elements using CSS animations instead of JS manipulation
    [...heartPoints, ...rayPoints].forEach((point, index) => {
      const element = document.createElement('div');
      element.className = 'neon-glow-particle';
      
      // Use CSS custom properties for easy updates
      element.style.cssText = `
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        left: ${point.x}px;
        top: ${point.y}px;
        transform: translate(-50%, -50%);
        background: transparent;
        box-shadow: 
          0 0 10px hsla(220, 90%, 60%, var(--neon-opacity)),
          0 0 20px hsla(260, 80%, 50%, calc(var(--neon-opacity) * 0.7)),
          0 0 30px hsla(220, 70%, 40%, calc(var(--neon-opacity) * 0.5));
        filter: blur(var(--neon-blur));
        animation: neonPulse var(--neon-speed) ease-in-out infinite;
        animation-delay: ${index * 50}ms;
        pointer-events: none;
      `;
      
      container.appendChild(element);
      elementsRef.current.push(element);
    });

    // Update CSS custom properties
    updateCSSProperties();
  };

  const updateCSSProperties = () => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    container.style.setProperty('--neon-opacity', options.opacity.toString());
    container.style.setProperty('--neon-blur', `${options.blur}px`);
    container.style.setProperty('--neon-speed', `${options.speed}ms`);
  };

  const start = () => {
    if (!options.enabled || isAnimating) return;
    
    setIsAnimating(true);
    createHeartChakra();
  };

  const stop = () => {
    setIsAnimating(false);
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    // Clean up elements
    elementsRef.current.forEach(el => {
      if (containerRef.current?.contains(el)) {
        containerRef.current.removeChild(el);
      }
    });
    elementsRef.current = [];
  };

  const updateOptions = (newOptions: Partial<NeonGlowOptions>) => {
    setOptions(prev => ({ ...prev, ...newOptions }));
  };

  // Add CSS keyframes when component mounts
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes neonPulse {
        0%, 100% {
          opacity: 1;
          transform: translate(-50%, -50%) scale(1);
        }
        50% {
          opacity: 0.6;
          transform: translate(-50%, -50%) scale(1.2);
        }
      }
      
      .neon-glow-container {
        position: relative;
        overflow: hidden;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, []);

  // Update CSS properties when options change
  useEffect(() => {
    if (isAnimating) {
      updateCSSProperties();
    }
  }, [options, isAnimating]);

  // Recreate pattern when size changes
  useEffect(() => {
    if (isAnimating && containerRef.current) {
      createHeartChakra();
    }
  }, [options.size]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, []);

  return {
    containerRef,
    isAnimating,
    start,
    stop,
    updateOptions
  };
}; 