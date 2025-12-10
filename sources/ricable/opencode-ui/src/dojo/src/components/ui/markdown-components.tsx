import React from "react";
import { cn } from "@/lib/utils";

export const MarkdownComponents = {
  // Header components
  h1: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => {
    return (
      <h1
        className={cn(
          "text-3xl font-bold mt-8 mb-6 text-foreground border-b border-border pb-2",
          className,
        )}
        {...props}
      >
        {children}
      </h1>
    );
  },

  h2: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => {
    return (
      <h2
        className={cn("text-2xl font-bold mt-8 mb-4 text-foreground", className)}
        {...props}
      >
        {children}
      </h2>
    );
  },

  h3: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => {
    return (
      <h3
        className={cn("text-xl font-bold mt-6 mb-3 text-foreground", className)}
        {...props}
      >
        {children}
      </h3>
    );
  },

  // Paragraph component
  p: ({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => {
    return (
      <p
        className={cn("my-4 text-muted-foreground leading-relaxed", className)}
        {...props}
      >
        {children}
      </p>
    );
  },

  // List components
  ul: ({ className, children, ...props }: React.HTMLAttributes<HTMLUListElement>) => {
    return (
      <ul className={cn("my-4 pl-6 list-disc space-y-2", className)} {...props}>
        {children}
      </ul>
    );
  },

  ol: ({ className, children, ...props }: React.HTMLAttributes<HTMLOListElement>) => {
    return (
      <ol className={cn("my-4 pl-6 list-decimal space-y-2", className)} {...props}>
        {children}
      </ol>
    );
  },

  li: ({ className, children, ...props }: React.HTMLAttributes<HTMLLIElement>) => {
    return (
      <li className={cn("text-muted-foreground my-1", className)} {...props}>
        {children}
      </li>
    );
  },

  // Custom code block rendering
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => {
    const match = /language-(\w+)/.exec(className || "");
    const language = match ? match[1] : "";

    // If it's an inline code block (no language specified and no line breaks)
    if (!match && typeof children === "string" && !children.includes("\n")) {
      return (
        <code
          className={cn(
            "bg-muted text-muted-foreground px-1.5 py-0.5 rounded text-sm font-mono",
            className,
          )}
          {...props}
        >
          {children}
        </code>
      );
    }

    return (
      <div className="relative group my-6">
        {language && (
          <div className="absolute right-2 top-2 text-xs text-muted-foreground font-mono bg-muted px-2 py-1 rounded">
            {language}
          </div>
        )}
        <pre
          className={cn(
            "p-4 rounded-lg overflow-x-auto border bg-muted text-muted-foreground",
            className,
          )}
        >
          <code {...props} className="text-sm font-mono">
            {children}
          </code>
        </pre>
      </div>
    );
  },

  // Custom link rendering
  a: ({ className, children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
    return (
      <a
        className={cn(
          "text-primary font-medium underline underline-offset-2 hover:text-primary/80 transition-colors",
          className,
        )}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    );
  },

  // Custom table rendering
  table: ({ className, children, ...props }: React.TableHTMLAttributes<HTMLTableElement>) => {
    return (
      <div className="overflow-x-auto my-6">
        <table
          className={cn(
            "w-full border-collapse border",
            className,
          )}
          {...props}
        >
          {children}
        </table>
      </div>
    );
  },

  // Custom image rendering
  img: ({ className, alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) => {
    return (
      <img
        className={cn("rounded-lg mx-auto my-6 max-w-full h-auto", className)}
        alt={alt || ""}
        {...props}
      />
    );
  },

  // Blockquote component
  blockquote: ({ className, children, ...props }: React.HTMLAttributes<HTMLQuoteElement>) => {
    return (
      <blockquote
        className={cn(
          "border-l-4 border-border pl-4 py-1 my-4 italic text-muted-foreground",
          className,
        )}
        {...props}
      >
        {children}
      </blockquote>
    );
  },

  // Horizontal rule
  hr: ({ className, ...props }: React.HTMLAttributes<HTMLHRElement>) => {
    return (
      <hr
        className={cn("my-8 border-t border-border", className)}
        {...props}
      />
    );
  },

  // Strong/bold text
  strong: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => {
    return (
      <strong className={cn("font-bold text-foreground", className)} {...props}>
        {children}
      </strong>
    );
  },

  // Emphasis/italic text
  em: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => {
    return (
      <em className={cn("italic text-foreground", className)} {...props}>
        {children}
      </em>
    );
  },
};
