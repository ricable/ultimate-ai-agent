/**
 * Problem Solving Tasks Benchmark Scenario
 *
 * Tests agent's ability to:
 * - Break down complex problems
 * - Apply algorithms and data structures
 * - Optimize for time and space complexity
 * - Reason about edge cases
 * - Transfer knowledge between domains
 */

import type { BenchmarkScenario, Task } from '../lib/types.js';

const tasks: Task[] = [
  {
    id: 'problem-01',
    name: 'Two Sum Problem',
    description: 'Find two numbers in an array that add up to a target value.',
    domain: 'problem-solving',
    input: `Given an array of integers and a target sum, return indices of two numbers that add up to target. Each input has exactly one solution.
Example: [2, 7, 11, 15], target = 9 → [0, 1] (because 2 + 7 = 9)`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('map') || code.includes('set') || code.includes('object')) &&
        code.includes('target') &&
        (code.includes('complement') || code.includes('diff'))
      );
    },
    difficulty: 'easy'
  },
  {
    id: 'problem-02',
    name: 'Valid Parentheses',
    description: 'Determine if a string of brackets is valid.',
    domain: 'problem-solving',
    input: `Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. Valid means every opening bracket has a corresponding closing bracket in the correct order.`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('stack') &&
        (code.includes('push') && code.includes('pop')) &&
        (code.includes('match') || code.includes('pair'))
      );
    },
    difficulty: 'easy'
  },
  {
    id: 'problem-03',
    name: 'Longest Substring Without Repeating',
    description: 'Find the length of the longest substring without repeating characters.',
    domain: 'problem-solving',
    input: `Given a string, find the length of the longest substring without repeating characters.
Example: "abcabcbb" → 3 (substring "abc")`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('set') || code.includes('map')) &&
        (code.includes('window') || code.includes('sliding')) &&
        (code.includes('max') || code.includes('longest'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'problem-04',
    name: 'Merge Intervals',
    description: 'Merge overlapping intervals in a collection.',
    domain: 'problem-solving',
    input: `Given a collection of intervals, merge all overlapping intervals.
Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('sort') &&
        (code.includes('overlap') || code.includes('merge')) &&
        (code.includes('start') && code.includes('end'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'problem-05',
    name: 'Binary Tree Level Order Traversal',
    description: 'Traverse a binary tree level by level.',
    domain: 'problem-solving',
    input: `Given a binary tree, return the level order traversal of its nodes' values (from left to right, level by level).`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('queue') &&
        (code.includes('level') || code.includes('bfs') || code.includes('breadth')) &&
        (code.includes('left') && code.includes('right'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'problem-06',
    name: 'Word Ladder',
    description: 'Find the shortest transformation sequence from one word to another.',
    domain: 'problem-solving',
    input: `Given two words (beginWord and endWord) and a dictionary, find the length of shortest transformation sequence from beginWord to endWord. Each transformed word must exist in dictionary, and only one letter can be changed at a time.`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('bfs') || code.includes('queue')) &&
        (code.includes('visited') || code.includes('seen')) &&
        (code.includes('neighbor') || code.includes('adjacent'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'problem-07',
    name: 'Coin Change',
    description: 'Find minimum number of coins needed to make up an amount.',
    domain: 'problem-solving',
    input: `Given coins of different denominations and a total amount, compute the fewest number of coins needed to make up that amount. If amount cannot be made up, return -1.`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('dp') || code.includes('dynamic')) &&
        (code.includes('min') || code.includes('minimum')) &&
        (code.includes('amount') || code.includes('sum'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'problem-08',
    name: 'Serialize and Deserialize Binary Tree',
    description: 'Design algorithm to serialize and deserialize a binary tree.',
    domain: 'problem-solving',
    input: `Design an algorithm to serialize a binary tree to a string and deserialize the string back to the original tree structure.`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        code.includes('serialize') &&
        code.includes('deserialize') &&
        (code.includes('null') || code.includes('none')) &&
        (code.includes('queue') || code.includes('stack') || code.includes('recursive'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'problem-09',
    name: 'Trapping Rain Water',
    description: 'Calculate how much water can be trapped after raining.',
    domain: 'problem-solving',
    input: `Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
Example: [0,1,0,2,1,0,1,3,2,1,2,1] → 6 units of water`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('left') && code.includes('right')) &&
        (code.includes('max') || code.includes('height')) &&
        (code.includes('water') || code.includes('trap'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'problem-10',
    name: 'Regular Expression Matching',
    description: 'Implement regular expression matching with . and * support.',
    domain: 'problem-solving',
    input: `Implement regular expression matching with support for '.' (matches any single character) and '*' (matches zero or more of the preceding element). The matching should cover the entire input string.`,
    successCriteria: (result) => {
      const code = result.output.toLowerCase();
      return (
        (code.includes('dp') || code.includes('dynamic') || code.includes('recursive')) &&
        (code.includes('match') || code.includes('pattern')) &&
        (code.includes('*') || code.includes('star')) &&
        (code.includes('.') || code.includes('dot') || code.includes('any'))
      );
    },
    difficulty: 'hard'
  }
];

export const problemSolvingTasksScenario: BenchmarkScenario = {
  name: 'problem-solving-tasks',
  description: 'Tests ability to solve algorithmic problems and apply data structures',
  domain: 'problem-solving',
  tasks,
  metrics: [
    'success_rate',
    'token_efficiency',
    'latency',
    'memory_usage',
    'learning_velocity',
    'accuracy'
  ]
};
