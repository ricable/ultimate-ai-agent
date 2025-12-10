import React, { useMemo, useState, createContext, useContext } from 'react';
interface TabsProps {
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}
const TabsContext = createContext<{
  value: string;
  onValueChange: (value: string) => void;
}>({
  value: '',
  onValueChange: () => {}
});
export const Tabs = ({
  defaultValue,
  value,
  onValueChange,
  children,
  className = ''
}: TabsProps) => {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const activeValue = value !== undefined ? value : internalValue;
  const contextValue = useMemo(() => ({
    value: activeValue,
    onValueChange: (newValue: string) => {
      setInternalValue(newValue);
      onValueChange?.(newValue);
    }
  }), [activeValue, onValueChange]);
  return <TabsContext.Provider value={contextValue}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>;
};
interface TabsListProps {
  children: React.ReactNode;
  className?: string;
}
export const TabsList = ({
  children,
  className = ''
}: TabsListProps) => {
  return <div className={`relative ${className}`} role="tablist">
      {/* Subtle neon glow effect */}
      <div className="absolute inset-0 rounded-lg opacity-30 blur-[1px] bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 pointer-events-none"></div>
      {children}
    </div>;
};
interface TabsTriggerProps {
  value: string;
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  color?: 'blue' | 'purple' | 'pink' | 'orange' | 'cyan' | 'green';
}
export const TabsTrigger = ({
  value,
  children,
  className = '',
  onClick,
  color = 'blue'
}: TabsTriggerProps) => {
  const {
    value: activeValue,
    onValueChange
  } = useContext(TabsContext);
  const isActive = activeValue === value;
  const handleClick = () => {
    onValueChange(value);
    onClick?.();
  };
  const colorMap = {
    blue: {
      text: 'text-blue-600 dark:text-blue-400',
      glow: 'bg-blue-500 shadow-[0_0_10px_2px_rgba(59,130,246,0.4)] dark:shadow-[0_0_20px_5px_rgba(59,130,246,0.7)]',
      hover: 'hover:text-blue-500 dark:hover:text-blue-400/70'
    },
    purple: {
      text: 'text-purple-600 dark:text-purple-400',
      glow: 'bg-purple-500 shadow-[0_0_10px_2px_rgba(168,85,247,0.4)] dark:shadow-[0_0_20px_5px_rgba(168,85,247,0.7)]',
      hover: 'hover:text-purple-500 dark:hover:text-purple-400/70'
    },
    pink: {
      text: 'text-pink-600 dark:text-pink-400',
      glow: 'bg-pink-500 shadow-[0_0_10px_2px_rgba(236,72,153,0.4)] dark:shadow-[0_0_20px_5px_rgba(236,72,153,0.7)]',
      hover: 'hover:text-pink-500 dark:hover:text-pink-400/70'
    },
    orange: {
      text: 'text-orange-600 dark:text-orange-400',
      glow: 'bg-orange-500 shadow-[0_0_10px_2px_rgba(249,115,22,0.4)] dark:shadow-[0_0_20px_5px_rgba(249,115,22,0.7)]',
      hover: 'hover:text-orange-500 dark:hover:text-orange-400/70'
    },
    cyan: {
      text: 'text-cyan-600 dark:text-cyan-400',
      glow: 'bg-cyan-500 shadow-[0_0_10px_2px_rgba(34,211,238,0.4)] dark:shadow-[0_0_20px_5px_rgba(34,211,238,0.7)]',
      hover: 'hover:text-cyan-500 dark:hover:text-cyan-400/70'
    },
    green: {
      text: 'text-emerald-600 dark:text-emerald-400',
      glow: 'bg-emerald-500 shadow-[0_0_10px_2px_rgba(16,185,129,0.4)] dark:shadow-[0_0_20px_5px_rgba(16,185,129,0.7)]',
      hover: 'hover:text-emerald-500 dark:hover:text-emerald-400/70'
    }
  };
  return <button className={`
        relative px-24 py-10 font-mono transition-all duration-300 z-10
        ${isActive ? colorMap[color].text : `text-gray-600 dark:text-gray-400 ${colorMap[color].hover}`}
        ${className}
      `} onClick={handleClick} type="button" role="tab" aria-selected={isActive} data-state={isActive ? 'active' : 'inactive'}>
      {children}
      {/* Active state neon indicator */}
      {isActive && <>
          <span className={`absolute bottom-0 left-0 right-0 w-full h-[2px] ${colorMap[color].glow}`}></span>
        </>}
    </button>;
};
interface TabsContentProps {
  value: string;
  children: React.ReactNode;
  className?: string;
}
export const TabsContent = ({
  value,
  children,
  className = ''
}: TabsContentProps) => {
  const {
    value: activeValue
  } = useContext(TabsContext);
  // Simplified TabsContent - we're handling animations in the parent component now
  if (activeValue !== value) return null;
  return <div className={className} role="tabpanel" data-state={activeValue === value ? 'active' : 'inactive'}>
      {children}
    </div>;
};