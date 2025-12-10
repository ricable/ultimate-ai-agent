#!/usr/bin/env node
const { JjWrapper } = require('./index.js');

const jj = new JjWrapper();
const proto = Object.getPrototypeOf(jj);
const methods = Object.getOwnPropertyNames(proto).filter(m => m !== 'constructor').sort();

console.log('Available methods in JjWrapper:');
console.log(methods.join('\n'));

console.log('\n\nChecking for ReasoningBank methods:');
const rbMethods = [
    'startTrajectory',
    'addToTrajectory',
    'finalizeTrajectory',
    'getSuggestion',
    'getLearningStats',
    'getPatterns',
    'queryTrajectories',
    'resetLearning'
];

rbMethods.forEach(method => {
    const exists = methods.includes(method);
    console.log(`  ${method}: ${exists ? '✅ EXISTS' : '❌ MISSING'}`);
});
