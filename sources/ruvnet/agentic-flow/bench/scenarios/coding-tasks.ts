/**
 * Coding Tasks Benchmark Scenario
 *
 * Tests agent's ability to:
 * - Implement functions from specifications
 * - Write clean, idiomatic code
 * - Handle edge cases
 * - Follow best practices
 */

import type { BenchmarkScenario, Task } from '../lib/types.js';

const tasks: Task[] = [
  {
    id: 'coding-01',
    name: 'Implement Array Deduplication',
    description: 'Write a function that removes duplicate elements from an array while preserving order.',
    domain: 'coding',
    input: `function deduplicate(arr: any[]): any[] {
  // TODO: implement
}`,
    expectedOutput: `function deduplicate(arr: any[]): any[] {
  return [...new Set(arr)];
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('set') ||
        (code.includes('filter') && code.includes('indexof'))
      ) && !result.output.includes('TODO');
    },
    difficulty: 'easy'
  },
  {
    id: 'coding-02',
    name: 'Implement Deep Clone',
    description: 'Write a function that creates a deep clone of an object, handling nested objects and arrays.',
    domain: 'coding',
    input: `function deepClone(obj: any): any {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('json.parse') && code.includes('json.stringify')) ||
        (code.includes('recursive') || code.includes('object.assign')) &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'coding-03',
    name: 'Implement Debounce Function',
    description: 'Write a debounce function that delays execution until after a specified wait time has elapsed.',
    domain: 'coding',
    input: `function debounce(func: Function, wait: number): Function {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('settimeout') &&
        code.includes('cleartimeout') &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'coding-04',
    name: 'Implement Promise Retry',
    description: 'Write a function that retries a promise-returning function up to N times with exponential backoff.',
    domain: 'coding',
    input: `async function retry<T>(fn: () => Promise<T>, maxRetries: number): Promise<T> {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('async') &&
        code.includes('await') &&
        code.includes('catch') &&
        (code.includes('for') || code.includes('while')) &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'coding-05',
    name: 'Implement LRU Cache',
    description: 'Write an LRU (Least Recently Used) cache with get and set operations.',
    domain: 'coding',
    input: `class LRUCache<K, V> {
  constructor(private capacity: number) {}
  get(key: K): V | undefined {}
  set(key: K, value: V): void {}
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('map') &&
        (code.includes('delete') || code.includes('remove')) &&
        code.includes('size') &&
        !result.output.includes('// TODO')
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'coding-06',
    name: 'Implement Binary Search',
    description: 'Write a binary search function that finds the index of a target in a sorted array.',
    domain: 'coding',
    input: `function binarySearch(arr: number[], target: number): number {
  // TODO: implement, return -1 if not found
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('while') || code.includes('recursive')) &&
        code.includes('mid') &&
        code.includes('left') &&
        code.includes('right') &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'coding-07',
    name: 'Implement Flatten Array',
    description: 'Write a function that flattens a nested array to any depth.',
    domain: 'coding',
    input: `function flatten(arr: any[]): any[] {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('flat') || code.includes('reduce') || code.includes('recursive')) &&
        (code.includes('array.isarray') || code.includes('concat')) &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'coding-08',
    name: 'Implement Throttle Function',
    description: 'Write a throttle function that limits function execution to once per specified interval.',
    domain: 'coding',
    input: `function throttle(func: Function, interval: number): Function {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('settimeout') &&
        (code.includes('last') || code.includes('time')) &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'coding-09',
    name: 'Implement Event Emitter',
    description: 'Write a simple event emitter class with on, off, and emit methods.',
    domain: 'coding',
    input: `class EventEmitter {
  on(event: string, handler: Function): void {}
  off(event: string, handler: Function): void {}
  emit(event: string, ...args: any[]): void {}
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('map') &&
        code.includes('set') &&
        code.includes('get') &&
        code.includes('foreach') &&
        !result.output.includes('// TODO')
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'coding-10',
    name: 'Implement Memoization',
    description: 'Write a memoization function that caches results of expensive function calls.',
    domain: 'coding',
    input: `function memoize<T extends (...args: any[]) => any>(fn: T): T {
  // TODO: implement
}`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('map') || code.includes('cache')) &&
        code.includes('has') &&
        code.includes('get') &&
        code.includes('set') &&
        !result.output.includes('TODO')
      );
    },
    difficulty: 'medium'
  }
];

export const codingTasksScenario: BenchmarkScenario = {
  name: 'coding-tasks',
  description: 'Tests ability to implement common programming patterns and algorithms',
  domain: 'coding',
  tasks,
  metrics: [
    'success_rate',
    'token_efficiency',
    'latency',
    'memory_usage',
    'learning_velocity'
  ]
};
