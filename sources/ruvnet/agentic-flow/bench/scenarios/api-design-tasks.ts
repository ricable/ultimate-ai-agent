/**
 * API Design Tasks Benchmark Scenario
 *
 * Tests agent's ability to:
 * - Design RESTful APIs
 * - Apply API best practices
 * - Handle versioning and deprecation
 * - Design authentication systems
 * - Structure request/response schemas
 */

import type { BenchmarkScenario, Task } from '../lib/types.js';

const tasks: Task[] = [
  {
    id: 'api-01',
    name: 'Design User Authentication API',
    description: 'Design a RESTful API for user authentication including signup, login, logout, and token refresh.',
    domain: 'api-design',
    input: `Design endpoints for user authentication system with JWT tokens.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('post') &&
        output.includes('/auth/signup') ||
        output.includes('/signup') &&
        output.includes('/login') &&
        output.includes('jwt') &&
        output.includes('refresh')
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'api-02',
    name: 'Design RESTful CRUD API',
    description: 'Design a RESTful API for managing blog posts with proper HTTP methods and status codes.',
    domain: 'api-design',
    input: `Design CRUD endpoints for blog posts including listing, creating, updating, and deleting.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('get') &&
        output.includes('post') &&
        output.includes('put') ||
        output.includes('patch') &&
        output.includes('delete') &&
        (output.includes('/posts') || output.includes('/articles'))
      );
    },
    difficulty: 'easy'
  },
  {
    id: 'api-03',
    name: 'Design Pagination API',
    description: 'Design pagination for a large dataset with cursor-based or offset-based pagination.',
    domain: 'api-design',
    input: `Design API pagination for listing products with 10,000+ items.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        (output.includes('cursor') || output.includes('offset')) &&
        (output.includes('limit') || output.includes('page')) &&
        (output.includes('next') || output.includes('has_more'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'api-04',
    name: 'Design Rate Limiting',
    description: 'Design rate limiting strategy for API with appropriate headers and error responses.',
    domain: 'api-design',
    input: `Design rate limiting for API with 100 requests per minute per user.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        (output.includes('x-ratelimit-limit') || output.includes('rate limit')) &&
        (output.includes('429') || output.includes('too many requests')) &&
        (output.includes('retry-after') || output.includes('reset'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'api-05',
    name: 'Design API Versioning',
    description: 'Design API versioning strategy to maintain backward compatibility.',
    domain: 'api-design',
    input: `Design API versioning for evolving endpoints without breaking existing clients.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        (output.includes('/v1') || output.includes('/v2')) ||
        (output.includes('accept-version') || output.includes('api-version')) &&
        (output.includes('deprecat') || output.includes('backward compat'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'api-06',
    name: 'Design Error Response Schema',
    description: 'Design consistent error response format with proper status codes and error details.',
    domain: 'api-design',
    input: `Design error response schema for API with validation errors, authorization errors, and server errors.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('error') &&
        (output.includes('message') || output.includes('detail')) &&
        (output.includes('code') || output.includes('type')) &&
        (output.includes('400') || output.includes('401') || output.includes('500'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'api-07',
    name: 'Design File Upload API',
    description: 'Design file upload endpoint with progress tracking and validation.',
    domain: 'api-design',
    input: `Design API for uploading large files (up to 100MB) with progress tracking.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('multipart') ||
        output.includes('form-data') &&
        (output.includes('chunk') || output.includes('stream')) &&
        (output.includes('progress') || output.includes('status'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'api-08',
    name: 'Design Search and Filter API',
    description: 'Design search and filtering API with multiple criteria and sorting.',
    domain: 'api-design',
    input: `Design search API for products with filtering by category, price range, and sorting options.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        (output.includes('q=') || output.includes('query=') || output.includes('search=')) &&
        (output.includes('filter') || output.includes('category')) &&
        (output.includes('sort') || output.includes('order'))
      );
    },
    difficulty: 'medium'
  },
  {
    id: 'api-09',
    name: 'Design Webhook API',
    description: 'Design webhook system for real-time event notifications.',
    domain: 'api-design',
    input: `Design webhook API for notifying clients of order status changes.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('webhook') &&
        (output.includes('event') || output.includes('payload')) &&
        (output.includes('signature') || output.includes('verify') || output.includes('hmac')) &&
        (output.includes('retry') || output.includes('callback'))
      );
    },
    difficulty: 'hard'
  },
  {
    id: 'api-10',
    name: 'Design GraphQL Schema',
    description: 'Design GraphQL schema for flexible data querying with types and resolvers.',
    domain: 'api-design',
    input: `Design GraphQL schema for blog system with posts, comments, and users.`,
    successCriteria: (result) => {
      const output = result.output.toLowerCase();
      return (
        output.includes('type') &&
        output.includes('query') &&
        (output.includes('mutation') || output.includes('resolver')) &&
        (output.includes('post') && output.includes('comment') && output.includes('user'))
      );
    },
    difficulty: 'hard'
  }
];

export const apiDesignTasksScenario: BenchmarkScenario = {
  name: 'api-design-tasks',
  description: 'Tests ability to design RESTful APIs and follow API best practices',
  domain: 'api-design',
  tasks,
  metrics: [
    'success_rate',
    'token_efficiency',
    'latency',
    'memory_usage',
    'learning_velocity'
  ]
};
