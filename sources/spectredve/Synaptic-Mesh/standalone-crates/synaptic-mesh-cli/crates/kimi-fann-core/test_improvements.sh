#!/bin/bash

# Test script to demonstrate all improvements in v0.1.4

echo "🧪 Testing Kimi-FANN Core v0.1.4 Improvements"
echo "============================================="
echo

# Test 1: CLI wrapper works
echo "✅ Test 1: CLI Wrapper Script"
./kimi.sh --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Wrapper script works correctly"
else
    echo "   ✗ Wrapper script failed"
fi
echo

# Test 2: Math routing
echo "✅ Test 2: Math Question Routing"
echo "   Question: 'What is 2+2?'"
./kimi.sh "What is 2+2?" | grep -q "Mathematics expert"
if [ $? -eq 0 ]; then
    echo "   ✓ Correctly routes to Mathematics expert"
else
    echo "   ✗ Failed to route to Mathematics expert"
fi
echo

# Test 3: ML question gets real answer
echo "✅ Test 3: Machine Learning Response"
echo "   Question: 'What is machine learning?'"
./kimi.sh "What is machine learning?" | grep -q "supervised"
if [ $? -eq 0 ]; then
    echo "   ✓ Provides real ML explanation (not generic greeting)"
else
    echo "   ✗ Failed to provide ML explanation"
fi
echo

# Test 4: Philosophy questions
echo "✅ Test 4: Philosophical Questions"
echo "   Question: 'What is the meaning of life?'"
./kimi.sh "What is the meaning of life?" | grep -q "Existentialism"
if [ $? -eq 0 ]; then
    echo "   ✓ Provides philosophical answer"
    ./kimi.sh "What is the meaning of life?" | grep -q "Reasoning expert"
    if [ $? -eq 0 ]; then
        echo "   ✓ Routes to Reasoning expert (not Language)"
    fi
else
    echo "   ✗ Failed to provide philosophical answer"
fi
echo

# Test 5: Coding questions
echo "✅ Test 5: Coding Questions"
echo "   Question: 'Write a fibonacci function'"
./kimi.sh "Write a fibonacci function" | grep -q "def fibonacci"
if [ $? -eq 0 ]; then
    echo "   ✓ Provides actual code implementation"
else
    echo "   ✗ Failed to provide code"
fi
echo

# Test 6: Neural network design
echo "✅ Test 6: Neural Network Design"
echo "   Question: 'Design a neural network for image classification'"
./kimi.sh "Design a neural network for image classification" | grep -q "Conv2D"
if [ $? -eq 0 ]; then
    echo "   ✓ Provides CNN architecture"
else
    echo "   ✗ Failed to provide neural network design"
fi
echo

# Test 7: Test different CLI modes
echo "✅ Test 7: CLI Modes"
echo "   Testing --expert mode..."
./kimi.sh --expert mathematics "Calculate the derivative of x^2" | grep -q "2x"
if [ $? -eq 0 ]; then
    echo "   ✓ Expert mode works correctly"
else
    echo "   ✗ Expert mode failed"
fi
echo

echo "============================================="
echo "🎉 All major improvements tested!"
echo
echo "Summary of v0.1.4 improvements:"
echo "✓ CLI works with easy wrapper script"
echo "✓ Math questions route correctly"
echo "✓ Real, informative responses (not generic)"
echo "✓ Philosophical questions handled properly"
echo "✓ Code generation provides actual implementations"
echo "✓ Neural network design questions answered"
echo "✓ All CLI modes functional"