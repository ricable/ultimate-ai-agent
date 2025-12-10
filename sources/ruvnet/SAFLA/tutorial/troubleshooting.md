# SAFLA Tutorial Troubleshooting Guide

## Common Issues and Solutions

This guide helps resolve common issues you might encounter while following the SAFLA Hello World tutorial.

## Mode Transition Issues

### Problem: Mode doesn't spawn the next task
**Symptoms:**
- Mode completes work but doesn't use `new_task`
- Workflow stops at a particular mode
- No clear handoff to the next phase

**Solutions:**
1. **Remind the mode of its completion responsibility:**
   ```
   Please complete this phase and spawn the next task using new_task: [mode-name]
   ```

2. **Check if the mode needs more context:**
   ```
   Based on the work completed, what should be the next step in the SAFLA workflow?
   ```

3. **Manually transition if needed:**
   ```
   switch_mode: [next-mode-slug]
   ```

### Problem: Wrong mode spawned
**Symptoms:**
- Mode spawns an unexpected next task
- Workflow jumps phases or goes backward
- Mode coordination seems confused

**Solutions:**
1. **Clarify the expected workflow:**
   ```
   The next phase should be [expected-mode]. Please spawn that task instead.
   ```

2. **Provide workflow context:**
   ```
   We're following the SAFLA Hello World tutorial workflow: 
   Architect → Code → TDD → Memory → Meta-Cognitive → Scorer → Critic → Final Assembly
   ```

## Mode Behavior Issues

### Problem: Mode doesn't follow specialization
**Symptoms:**
- System Architect tries to write code
- Auto-Coder attempts system design
- Modes work outside their expertise

**Solutions:**
1. **Remind mode of its role:**
   ```
   As [Mode Name], please focus on [specific responsibility] and avoid [out-of-scope activities].
   ```

2. **Reference the workflow guide:**
   ```
   Please refer to the workflow-reference.md for your mode's specific responsibilities.
   ```

### Problem: Mode produces incomplete work
**Symptoms:**
- Missing key components or files
- Incomplete implementation
- Partial analysis or documentation

**Solutions:**
1. **Request completion:**
   ```
   Please complete all components mentioned in your initial plan before proceeding.
   ```

2. **Specify missing elements:**
   ```
   The following items appear to be missing: [list specific items]
   ```

3. **Ask for verification:**
   ```
   Please verify that all deliverables for this phase are complete.
   ```

## Implementation Issues

### Problem: Code doesn't follow architecture
**Symptoms:**
- Implementation differs from architectural design
- Missing components or interfaces
- Inconsistent with specifications

**Solutions:**
1. **Reference the architecture:**
   ```
   Please ensure the implementation follows the architecture designed in the System Architect phase.
   ```

2. **Request alignment check:**
   ```
   Please verify that this implementation matches the original architectural specifications.
   ```

### Problem: Tests don't match implementation
**Symptoms:**
- Test failures due to interface mismatches
- Tests for non-existent functionality
- Missing tests for implemented features

**Solutions:**
1. **Request test-code alignment:**
   ```
   Please ensure tests match the actual implementation from the Auto-Coder phase.
   ```

2. **Provide implementation context:**
   ```
   Here's the actual implementation structure: [provide file list or key interfaces]
   ```

## Memory and Learning Issues

### Problem: Memory integration seems superficial
**Symptoms:**
- Memory components don't integrate with core functionality
- No clear learning or adaptation mechanisms
- Memory appears disconnected from the greeting system

**Solutions:**
1. **Request deeper integration:**
   ```
   Please ensure memory components actively enhance the greeting functionality with learning and personalization.
   ```

2. **Specify integration points:**
   ```
   Show how vector memory, episodic memory, and semantic memory specifically improve the Hello World system.
   ```

### Problem: Meta-cognitive features unclear
**Symptoms:**
- Self-awareness components seem abstract
- No clear self-monitoring or adaptation
- Meta-cognitive features don't connect to system behavior

**Solutions:**
1. **Request concrete examples:**
   ```
   Please provide specific examples of how the system monitors and improves its own performance.
   ```

2. **Ask for demonstration:**
   ```
   Show how the meta-cognitive capabilities would work in practice with the greeting system.
   ```

## Performance and Quality Issues

### Problem: Performance analysis lacks depth
**Symptoms:**
- Generic performance metrics
- No specific benchmarks or targets
- Unclear optimization recommendations

**Solutions:**
1. **Request specific metrics:**
   ```
   Please provide concrete performance benchmarks and specific optimization recommendations for the Hello World system.
   ```

2. **Ask for measurement details:**
   ```
   How would you measure and track the performance improvements over time?
   ```

### Problem: Code review misses key issues
**Symptoms:**
- Superficial code analysis
- Missing security or maintainability concerns
- No actionable improvement recommendations

**Solutions:**
1. **Request deeper analysis:**
   ```
   Please provide a more detailed code review focusing on security, maintainability, and architectural quality.
   ```

2. **Specify review criteria:**
   ```
   Please evaluate the code against SOLID principles, security best practices, and TypeScript conventions.
   ```

## Documentation and Integration Issues

### Problem: Documentation incomplete or unclear
**Symptoms:**
- Missing setup instructions
- Unclear usage examples
- Incomplete API documentation

**Solutions:**
1. **Request comprehensive documentation:**
   ```
   Please ensure documentation includes setup, usage examples, and complete API reference.
   ```

2. **Ask for user perspective:**
   ```
   Write documentation from the perspective of someone who has never seen this system before.
   ```

### Problem: Final integration issues
**Symptoms:**
- Components don't work together
- Missing integration points
- Deployment problems

**Solutions:**
1. **Request integration testing:**
   ```
   Please verify that all components work together as an integrated system.
   ```

2. **Ask for end-to-end validation:**
   ```
   Demonstrate the complete workflow from user input to system response, including all SAFLA capabilities.
   ```

## General Workflow Issues

### Problem: Tutorial feels too abstract
**Symptoms:**
- Concepts explained but not demonstrated
- Missing concrete examples
- Unclear practical applications

**Solutions:**
1. **Request concrete examples:**
   ```
   Please provide specific, runnable examples that demonstrate these concepts in action.
   ```

2. **Ask for practical demonstration:**
   ```
   Show exactly how a user would interact with this system and see the SAFLA capabilities working.
   ```

### Problem: Learning outcomes unclear
**Symptoms:**
- Unclear what was accomplished
- Missing connection to SAFLA principles
- No clear takeaways

**Solutions:**
1. **Request explicit connections:**
   ```
   Please explain how this work demonstrates specific SAFLA principles and capabilities.
   ```

2. **Ask for learning summary:**
   ```
   What should someone learn about SAFLA from this phase of the tutorial?
   ```

## Recovery Strategies

### When workflow gets stuck:
1. **Reset context:**
   ```
   Let's refocus on the SAFLA Hello World tutorial. We're currently in [current phase] and need to [specific objective].
   ```

2. **Provide clear direction:**
   ```
   Please complete [specific deliverable] and then spawn the next task for [next mode].
   ```

3. **Reference tutorial materials:**
   ```
   Please refer to tutorial/workflow-reference.md for guidance on this phase.
   ```

### When quality is insufficient:
1. **Set clear standards:**
   ```
   This work should demonstrate SAFLA principles with concrete, runnable examples that show adaptive learning and meta-cognitive capabilities.
   ```

2. **Request iteration:**
   ```
   Please enhance this work to better demonstrate [specific SAFLA concept] with practical examples.
   ```

### When coordination fails:
1. **Clarify workflow position:**
   ```
   We're following the 8-phase SAFLA tutorial. Current phase: [X]. Next phase: [Y].
   ```

2. **Provide workflow context:**
   ```
   Each phase should build on previous work while adding new SAFLA capabilities. Please ensure continuity.
   ```

## Success Indicators

### Good mode behavior:
- ✅ Stays within mode specialization
- ✅ Builds on previous work
- ✅ Demonstrates SAFLA principles
- ✅ Provides concrete, runnable examples
- ✅ Spawns appropriate next task

### Good workflow progression:
- ✅ Clear handoffs between modes
- ✅ Cumulative capability building
- ✅ Consistent architecture and implementation
- ✅ Progressive sophistication
- ✅ Clear learning outcomes

### Good SAFLA demonstration:
- ✅ Adaptive behavior examples
- ✅ Memory integration working
- ✅ Meta-cognitive capabilities visible
- ✅ Self-improvement mechanisms
- ✅ Emergent intelligence from coordination

Remember: The goal is to experience SAFLA principles through natural mode coordination, not just read about them. Each phase should add tangible capabilities that demonstrate adaptive learning and meta-cognitive intelligence.