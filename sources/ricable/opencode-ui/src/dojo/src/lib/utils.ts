import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Safely format a number to fixed decimal places
 * @param value - The number to format (can be null/undefined)
 * @param decimals - Number of decimal places (default: 2)
 * @param fallback - Fallback value if number is null/undefined (default: "0.00")
 */
export function safeToFixed(value: number | null | undefined, decimals: number = 2, fallback: string = "0.00"): string {
  if (value == null || isNaN(value)) {
    return fallback;
  }
  return value.toFixed(decimals);
}
