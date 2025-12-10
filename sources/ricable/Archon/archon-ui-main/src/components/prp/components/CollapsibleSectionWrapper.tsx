import React, { useState, useEffect, ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

interface CollapsibleSectionWrapperProps {
  children: ReactNode;
  header: ReactNode;
  isCollapsible?: boolean;
  defaultOpen?: boolean;
  isOpen?: boolean;
  onToggle?: () => void;
}

/**
 * A wrapper component that makes any section collapsible by clicking on its header
 */
export const CollapsibleSectionWrapper: React.FC<CollapsibleSectionWrapperProps> = ({
  children,
  header,
  isCollapsible = true,
  defaultOpen = true,
  isOpen: controlledIsOpen,
  onToggle
}) => {
  // Use controlled state if provided, otherwise manage internally
  const [internalIsOpen, setInternalIsOpen] = useState(defaultOpen);
  const isOpen = controlledIsOpen !== undefined ? controlledIsOpen : internalIsOpen;

  useEffect(() => {
    if (controlledIsOpen === undefined) {
      setInternalIsOpen(defaultOpen);
    }
  }, [defaultOpen, controlledIsOpen]);

  const handleToggle = () => {
    if (controlledIsOpen === undefined) {
      setInternalIsOpen(!internalIsOpen);
    }
    onToggle?.();
  };

  if (!isCollapsible) {
    return (
      <>
        {header}
        {children}
      </>
    );
  }

  return (
    <>
      <div 
        className="cursor-pointer select-none group"
        onClick={handleToggle}
      >
        <div className="relative">
          {header}
          <div className={`
            absolute right-4 top-1/2 -translate-y-1/2 
            transform transition-transform duration-200 
            ${isOpen ? 'rotate-180' : ''}
            text-gray-500 dark:text-gray-400
            group-hover:text-gray-700 dark:group-hover:text-gray-200
          `}>
            <ChevronDown className="w-5 h-5" />
          </div>
        </div>
      </div>
      
      <div className={`
        transition-all duration-300 
        ${isOpen ? 'max-h-none opacity-100' : 'max-h-0 opacity-0 overflow-hidden'}
      `}>
        <div className={isOpen ? 'pb-4' : ''}>
          {children}
        </div>
      </div>
    </>
  );
};