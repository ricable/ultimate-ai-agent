import { normalizeImagePlaceholders } from './normalizer';

/**
 * Formats a key into a human-readable label
 */
export function formatKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Truncates text with ellipsis
 */
export function truncateText(text: string, maxLength: number = 100): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

/**
 * Formats a value for display
 */
export function formatValue(value: any): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return value.toLocaleString();
  if (typeof value === 'string') {
    // Temporarily disabled to debug black screen issue
    // return normalizeImagePlaceholders(value);
    return value;
  }
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === 'object') return `${Object.keys(value).length} properties`;
  return String(value);
}

/**
 * Gets accent color based on index for variety
 */
export function getAccentColor(index: number): string {
  const colors = ['blue', 'purple', 'green', 'orange', 'pink', 'cyan', 'indigo', 'emerald'];
  return colors[index % colors.length];
}

/**
 * Generates a unique key for React components
 */
export function generateKey(prefix: string, ...parts: (string | number)[]): string {
  return [prefix, ...parts].filter(Boolean).join('-');
}