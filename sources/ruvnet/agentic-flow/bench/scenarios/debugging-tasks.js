/**
 * Debugging Tasks Benchmark Scenario
 *
 * Tests agent's ability to:
 * - Identify bugs in code
 * - Fix runtime and logical errors
 * - Handle edge cases
 * - Debug async code
 */
const tasks = [
    {
        id: 'debug-01',
        name: 'Fix Off-by-One Error',
        description: 'Find and fix the bug in this array iteration function that causes an index out of bounds error.',
        domain: 'debugging',
        input: `function sumArray(arr: number[]): number {
  let sum = 0;
  for (let i = 0; i <= arr.length; i++) {
    sum += arr[i];
  }
  return sum;
}`,
        successCriteria: (result) => {
            const code = result.output;
            return code.includes('i < arr.length') && !code.includes('i <= arr.length');
        },
        difficulty: 'easy'
    },
    {
        id: 'debug-02',
        name: 'Fix Async Race Condition',
        description: 'Fix the race condition in this async function that sometimes returns incorrect results.',
        domain: 'debugging',
        input: `let counter = 0;
async function increment(): Promise<number> {
  const current = counter;
  await new Promise(resolve => setTimeout(resolve, 10));
  counter = current + 1;
  return counter;
}`,
        successCriteria: (result) => {
            const code = result.output.toLowerCase();
            return (code.includes('lock') ||
                code.includes('mutex') ||
                code.includes('queue') ||
                !code.includes('const current = counter'));
        },
        difficulty: 'hard'
    },
    {
        id: 'debug-03',
        name: 'Fix Memory Leak',
        description: 'Identify and fix the memory leak in this event listener setup.',
        domain: 'debugging',
        input: `class Component {
  private listeners: Function[] = [];

  addEventListener(handler: Function) {
    this.listeners.push(handler);
    document.addEventListener('click', handler);
  }

  destroy() {
    this.listeners = [];
  }
}`,
        successCriteria: (result) => {
            const code = result.output;
            return code.includes('removeEventListener') && code.includes('destroy');
        },
        difficulty: 'medium'
    },
    {
        id: 'debug-04',
        name: 'Fix Type Coercion Bug',
        description: 'Fix the bug that causes incorrect comparisons due to type coercion.',
        domain: 'debugging',
        input: `function isEqual(a: any, b: any): boolean {
  return a == b;
}
// Bug: isEqual(0, "") returns true when it should return false`,
        successCriteria: (result) => {
            const code = result.output;
            return code.includes('===') && !code.includes('a == b');
        },
        difficulty: 'easy'
    },
    {
        id: 'debug-05',
        name: 'Fix Closure Issue',
        description: 'Fix the closure bug that causes all event handlers to reference the same variable.',
        domain: 'debugging',
        input: `function setupHandlers() {
  for (var i = 0; i < 5; i++) {
    document.getElementById('btn-' + i)?.addEventListener('click', () => {
      console.log('Button ' + i + ' clicked');
    });
  }
}`,
        successCriteria: (result) => {
            const code = result.output;
            return (code.includes('let i = 0') ||
                code.includes('const i') ||
                code.includes('((i) =>') ||
                code.includes('.bind('));
        },
        difficulty: 'medium'
    },
    {
        id: 'debug-06',
        name: 'Fix Promise Chain Error',
        description: 'Fix the error handling in this promise chain that swallows errors.',
        domain: 'debugging',
        input: `async function processData(data: string) {
  return fetch(data)
    .then(res => res.json())
    .then(json => json.value)
    .catch(err => console.error(err));
}`,
        successCriteria: (result) => {
            const code = result.output;
            return (code.includes('throw') ||
                code.includes('return Promise.reject') ||
                code.includes('async/await'));
        },
        difficulty: 'medium'
    },
    {
        id: 'debug-07',
        name: 'Fix Null Reference Error',
        description: 'Fix the code to handle null/undefined values properly.',
        domain: 'debugging',
        input: `function getUserName(user: { name?: string }): string {
  return user.name.toUpperCase();
}`,
        successCriteria: (result) => {
            const code = result.output;
            return (code.includes('?') ||
                code.includes('||') ||
                code.includes('if') ||
                code.includes('??'));
        },
        difficulty: 'easy'
    },
    {
        id: 'debug-08',
        name: 'Fix Stack Overflow',
        description: 'Fix the recursive function that causes stack overflow for large inputs.',
        domain: 'debugging',
        input: `function factorial(n: number): number {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}`,
        successCriteria: (result) => {
            const code = result.output.toLowerCase();
            return (code.includes('while') ||
                code.includes('for') ||
                code.includes('tail') ||
                code.includes('iterative'));
        },
        difficulty: 'medium'
    },
    {
        id: 'debug-09',
        name: 'Fix Infinite Loop',
        description: 'Fix the infinite loop bug in this array processing function.',
        domain: 'debugging',
        input: `function processArray(arr: number[]): number[] {
  let i = 0;
  while (i < arr.length) {
    if (arr[i] < 0) {
      arr.splice(i, 1);
    }
    i++;
  }
  return arr;
}`,
        successCriteria: (result) => {
            const code = result.output;
            return (code.includes('i++') && code.includes('else') ||
                code.includes('filter') ||
                !code.includes('i++'));
        },
        difficulty: 'medium'
    },
    {
        id: 'debug-10',
        name: 'Fix State Mutation Bug',
        description: 'Fix the bug where the function mutates the input instead of returning a new object.',
        domain: 'debugging',
        input: `function updateUser(user: { name: string; age: number }, updates: Partial<{ name: string; age: number }>): { name: string; age: number } {
  Object.assign(user, updates);
  return user;
}`,
        successCriteria: (result) => {
            const code = result.output;
            return (code.includes('...') && code.includes('user') ||
                code.includes('Object.assign({}, user') ||
                code.includes('structuredClone'));
        },
        difficulty: 'medium'
    }
];
export const debuggingTasksScenario = {
    name: 'debugging-tasks',
    description: 'Tests ability to identify and fix common programming bugs',
    domain: 'debugging',
    tasks,
    metrics: [
        'success_rate',
        'token_efficiency',
        'latency',
        'memory_usage',
        'accuracy'
    ]
};
