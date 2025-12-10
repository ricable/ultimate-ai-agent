/**
 * Simple calculator module with basic arithmetic operations
 */

/**
 * Add two numbers
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Sum of a and b
 */
export function add(a, b) {
  return a + b;
}

/**
 * Subtract two numbers
 * @param {number} a - First number
 * @param {number} b - Second number to subtract from a
 * @returns {number} Difference of a and b
 */
export function subtract(a, b) {
  return a - b;
}

// Default export with both functions
export default {
  add,
  subtract
};