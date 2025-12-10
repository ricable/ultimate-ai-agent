// neural-example.js - Example of using ruv-FANN neural networks via WASM

import { 
  createNeuralNetwork, 
  createTrainer, 
  createAgentNeuralManager,
  ActivationFunctions,
  ACTIVATION_FUNCTIONS,
  TRAINING_ALGORITHMS,
  COGNITIVE_PATTERNS,
  initializeNeuralWasm
} from '../src/neural-network.js';

// Example 1: Basic XOR neural network
async function xorExample() {
  console.log('=== XOR Neural Network Example ===');
  
  // Create a neural network for XOR problem
  const network = await createNeuralNetwork({
    inputSize: 2,
    hiddenLayers: [
      { size: 4, activation: ACTIVATION_FUNCTIONS.SIGMOID },
      { size: 3, activation: ACTIVATION_FUNCTIONS.SIGMOID }
    ],
    outputSize: 1,
    outputActivation: ACTIVATION_FUNCTIONS.SIGMOID
  });
  
  // Training data for XOR
  const trainingData = {
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    outputs: [[0], [1], [1], [0]]
  };
  
  // Create trainer
  const trainer = await createTrainer({
    algorithm: TRAINING_ALGORITHMS.RPROP,
    maxEpochs: 1000,
    targetError: 0.001
  });
  
  // Train the network
  console.log('Training XOR network...');
  const result = await trainer.trainUntilTarget(network, trainingData, 0.001, 1000);
  console.log('Training result:', result);
  
  // Test the network
  console.log('\nTesting XOR network:');
  for (const input of trainingData.inputs) {
    const output = await network.run(input);
    console.log(`Input: [${input}] => Output: ${output[0].toFixed(4)}`);
  }
  
  // Get network info
  console.log('\nNetwork info:', network.getInfo());
}

// Example 2: Multi-agent neural networks with cognitive patterns
async function multiAgentExample() {
  console.log('\n=== Multi-Agent Neural Network Example ===');
  
  const manager = await createAgentNeuralManager();
  
  // Create agents with different cognitive patterns
  const agents = [
    {
      agentId: 'researcher-001',
      agentType: 'researcher',
      cognitivePattern: COGNITIVE_PATTERNS.DIVERGENT,
      inputSize: 10,
      outputSize: 5,
      taskSpecialization: ['data_analysis', 'pattern_recognition']
    },
    {
      agentId: 'coder-001',
      agentType: 'coder',
      cognitivePattern: COGNITIVE_PATTERNS.CONVERGENT,
      inputSize: 8,
      outputSize: 3,
      taskSpecialization: ['code_generation', 'optimization']
    },
    {
      agentId: 'analyst-001',
      agentType: 'analyst',
      cognitivePattern: COGNITIVE_PATTERNS.CRITICAL,
      inputSize: 12,
      outputSize: 4,
      taskSpecialization: ['risk_assessment', 'decision_making']
    }
  ];
  
  // Create neural networks for each agent
  for (const agentConfig of agents) {
    const agentId = await manager.createAgentNetwork(agentConfig);
    console.log(`Created neural network for agent: ${agentId}`);
    
    // Get cognitive state
    const state = await manager.getAgentCognitiveState(agentId);
    console.log(`Cognitive state for ${agentId}:`, {
      pattern: state.cognitivePattern.pattern_type,
      architecture: state.neuralArchitecture,
      performance: state.performance
    });
  }
  
  // Simulate training for one agent
  console.log('\nTraining researcher agent...');
  const trainingData = {
    inputs: Array(50).fill(null).map(() => 
      Array(10).fill(null).map(() => Math.random())
    ),
    outputs: Array(50).fill(null).map(() => 
      Array(5).fill(null).map(() => Math.random() > 0.5 ? 1 : 0)
    )
  };
  
  const trainingResult = await manager.trainAgentNetwork('researcher-001', trainingData);
  console.log('Training result:', trainingResult);
  
  // Test inference
  const testInput = Array(10).fill(null).map(() => Math.random());
  const output = await manager.getAgentInference('researcher-001', testInput);
  console.log('Inference output:', output);
  
  // Simulate online learning during execution
  console.log('\nSimulating online adaptation...');
  const experienceData = {
    inputs: [testInput],
    expected_outputs: [output.map(v => v > 0.5 ? 1 : 0)],
    actual_outputs: [output],
    rewards: [0.8],
    context: { task: 'pattern_recognition', difficulty: 'medium' }
  };
  
  const adaptationResult = await manager.fineTuneDuringExecution('researcher-001', experienceData);
  console.log('Adaptation result:', adaptationResult);
}

// Example 3: Testing all activation functions
async function activationFunctionExample() {
  console.log('\n=== Activation Function Example ===');
  
  const wasm = await initializeNeuralWasm();
  
  // Get all activation functions
  const allFunctions = await ActivationFunctions.getAll(wasm);
  console.log('Available activation functions:');
  allFunctions.forEach(([name, description]) => {
    console.log(`  ${name}: ${description}`);
  });
  
  // Compare activation functions for different inputs
  console.log('\nComparing activation functions:');
  const testInputs = [-2, -1, -0.5, 0, 0.5, 1, 2];
  
  for (const input of testInputs) {
    const results = await ActivationFunctions.compare(wasm, input);
    console.log(`\nInput: ${input}`);
    Object.entries(results).forEach(([func, output]) => {
      console.log(`  ${func}: ${output.toFixed(4)}`);
    });
  }
  
  // Test specific activation function properties
  console.log('\nActivation function properties:');
  for (const funcName of ['sigmoid', 'relu', 'tanh', 'gaussian']) {
    const props = await ActivationFunctions.getProperties(wasm, funcName);
    console.log(`${funcName}:`, props);
  }
}

// Example 4: Cascade correlation for dynamic network growth
async function cascadeExample() {
  console.log('\n=== Cascade Correlation Example ===');
  
  // Create initial small network
  const network = await createNeuralNetwork({
    inputSize: 2,
    hiddenLayers: [], // Start with no hidden layers
    outputSize: 1,
    outputActivation: ACTIVATION_FUNCTIONS.SIGMOID
  });
  
  // Training data (more complex than XOR)
  const trainingData = {
    inputs: Array(20).fill(null).map(() => [Math.random(), Math.random()]),
    outputs: Array(20).fill(null).map((_, i) => {
      const [x, y] = trainingData.inputs[i];
      // Complex function: sin(x*pi) + cos(y*pi)
      return [(Math.sin(x * Math.PI) + Math.cos(y * Math.PI)) / 2];
    })
  };
  
  // Note: Cascade training would grow the network dynamically
  console.log('Initial network info:', network.getInfo());
  
  // Would use cascade trainer here if implemented
  // const cascadeTrainer = new CascadeTrainer(wasm, null, network, trainingData);
  // const result = await cascadeTrainer.train();
  // console.log('Cascade training result:', result);
}

// Run all examples
async function runAllExamples() {
  try {
    await xorExample();
    await multiAgentExample();
    await activationFunctionExample();
    await cascadeExample();
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Run if called directly
if (require.main === module) {
  runAllExamples();
}

module.exports = { xorExample, multiAgentExample, activationFunctionExample, cascadeExample };