# Improving Chat Quality with Prompt Engineering

This document outlines techniques for improving the quality of chat interactions with LLMs running on Apple Silicon hardware using llama.cpp and MLX frameworks.

## Table of Contents
1. [Understanding Prompt Engineering](#understanding-prompt-engineering)
2. [System Message Optimization](#system-message-optimization)
3. [Conversation Templates](#conversation-templates)
4. [Chain-of-Thought Techniques](#chain-of-thought-techniques)
5. [Few-Shot Learning](#few-shot-learning)
6. [Parameter Tuning](#parameter-tuning)
7. [Context Management](#context-management)
8. [Framework-Specific Optimizations](#framework-specific-optimizations)

## Understanding Prompt Engineering

Prompt engineering is the practice of designing inputs to language models to elicit the most useful, relevant, and accurate outputs. When running models locally on Apple Silicon, effective prompt engineering becomes even more critical as it can:

- Improve response quality without changing the model
- Reduce token usage, enabling longer conversations
- Compensate for limitations in smaller quantized models
- Guide the model toward specific response formats
- Prevent common issues like hallucinations or off-topic responses

## System Message Optimization

The system message sets the foundation for the entire conversation and defines the assistant's behavior.

### Best Practices

1. **Be specific about the assistant's role**:
   ```
   You are an expert Python programmer with deep knowledge of machine learning frameworks. You specialize in optimizing code for Apple Silicon hardware.
   ```

2. **Define interaction parameters**:
   ```
   Keep your answers concise and focused. Provide code examples when relevant. If you're unsure about something, acknowledge it rather than guessing.
   ```

3. **Set formatting expectations**:
   ```
   For code examples, always specify the language using Markdown syntax. Structure complex explanations with headings and bullet points.
   ```

4. **Specify knowledge boundaries**:
   ```
   Your knowledge cutoff is September 2021. For questions about newer technologies, clarify this limitation before attempting to answer.
   ```

### Examples

**Basic System Message:**
```
You are a helpful assistant.
```

**Optimized System Message:**
```
You are a helpful AI assistant with expertise in machine learning, programming, and Apple hardware. You provide clear, accurate information with appropriate detail based on the complexity of the question. For code, you use best practices and include comments. You acknowledge limitations in your knowledge rather than guessing. Format responses using Markdown for readability, with headings for sections and code blocks for examples.
```

## Conversation Templates

Different models work best with specific conversation formats. Our interfaces support multiple templates:

### ChatML Template (Default)

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello, who are you?
<|im_end|>
<|im_start|>assistant
I'm an AI assistant running on your Mac using Apple Silicon.
<|im_end|>
```

### Llama-2 Template

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello, who are you? [/INST] I'm an AI assistant running on your Mac using Apple Silicon. </s>
```

### Alpaca Template

```
### Instruction:
You are a helpful assistant.

### Input:
Hello, who are you?

### Response:
I'm an AI assistant running on your Mac using Apple Silicon.
```

### Best Practices

- Use ChatML for most modern models
- Use Llama-2 format specifically for Llama-2 family models
- Test different templates to see which works best for your specific model
- Be consistent with the template throughout the conversation

## Chain-of-Thought Techniques

Chain-of-Thought (CoT) prompting improves reasoning for complex questions by encouraging the model to break down its thinking process.

### Examples

**Standard Question:**
```
What is the result of (17 × 28) - (14 × 32)?
```

**Chain-of-Thought Prompt:**
```
Solve this step-by-step:
1) Calculate (17 × 28)
2) Calculate (14 × 32)
3) Subtract the second result from the first
What is the result of (17 × 28) - (14 × 32)?
```

### Self-Consistency CoT

For complex reasoning, have the model generate multiple solution paths:

```
Solve this problem using three different approaches, then determine which answer is correct:
What is the result of (17 × 28) - (14 × 32)?
```

### Best Practices

- Use CoT for math, logic, or multi-step reasoning problems
- Explicitly ask for step-by-step thinking
- Consider the token budget when using CoT (it uses more tokens)
- For quantized models (especially 4-bit), CoT becomes even more important

## Few-Shot Learning

Few-shot learning provides examples of desired inputs and outputs to guide the model's responses.

### Examples

**Zero-shot:**
```
Classify the following text as positive, negative, or neutral: "The service was quick but the food was cold."
```

**Few-shot:**
```
Classify the following texts as positive, negative, or neutral:

Text: "The movie was fantastic and I loved every minute of it."
Classification: positive

Text: "The restaurant was too expensive and the portions were small."
Classification: negative

Text: "The package arrived on the expected delivery date."
Classification: neutral

Text: "The service was quick but the food was cold."
Classification:
```

### Best Practices

- Provide 2-5 examples for best results
- Make examples diverse but representative
- Order matters - put the most relevant examples last
- Match the format exactly between examples and the target question
- For format-sensitive tasks (like JSON output), always use few-shot examples

## Parameter Tuning

Tuning generation parameters can significantly improve chat quality.

### Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| Temperature | Controls randomness (0.0-2.0) | 0.7 for balanced responses<br>0.2 for factual/coding<br>1.0+ for creative tasks |
| Max Tokens | Maximum response length | 1024 for general use<br>256 for concise answers<br>2048+ for detailed explanations |
| Top-p | Nucleus sampling threshold | 0.9 for general use |
| Repetition Penalty | Penalizes repetitive text | 1.1 for general use<br>1.2+ to avoid repetition |

### Best Practices

- Lower temperature (0.1-0.3) for factual questions, coding, or specific formats
- Higher temperature (0.7-1.0) for creative writing or brainstorming
- Adjust max tokens based on expected response length
- Increase repetition penalty if you notice the model repeating phrases

## Context Management

Efficiently managing context is crucial for maintaining conversation quality, especially with limited context windows on local models.

### Best Practices

1. **Summarize periodically**:
   ```
   Please summarize our conversation so far, then continue helping me with my Python project.
   ```

2. **Focus the context**:
   ```
   Let's focus on the API design part of our discussion and continue from there.
   ```

3. **Manage token usage**:
   - Remove irrelevant parts of conversation
   - Keep system message concise
   - For lengthy code examples, extract the relevant parts

4. **Context window optimization**:
   - 7B models with 4-bit quantization: limit to ~2048 tokens
   - 13B models with 4-bit quantization: limit to ~4096 tokens
   - Larger context requires more memory and may slow down generation

## Framework-Specific Optimizations

### llama.cpp Optimizations

1. **Memory efficiency**:
   - Use `--mlock` to prevent memory swapping
   - Use `--ctx-size` to adjust context window as needed

2. **Performance**:
   - Enable Metal with `--metal` flag
   - Use `--threads` to match your CPU core count
   - Try `--batch-size` of 512 for bulk generation

3. **Quality settings**:
   ```bash
   ./main -m model.gguf --metal -t 4 \
     --temp 0.7 \
     --top-p 0.9 \
     --repeat-penalty 1.1 \
     -c 2048
   ```

### MLX Optimizations

1. **Memory efficiency**:
   - Use 4-bit quantization for most models
   - Manage Python memory with careful tensor handling

2. **Performance**:
   ```python
   import mlx.core as mx
   mx.set_default_device(mx.gpu)  # Ensure using GPU
   ```

3. **Quality settings**:
   ```python
   tokens = generate(
       model,
       tokenizer,
       prompt,
       max_tokens=1024,
       temp=0.7,
       top_p=0.95,
       repetition_penalty=1.1
   )
   ```

## Conclusion

Effective prompt engineering and parameter tuning can dramatically improve the quality of interactions with locally-run LLMs on Apple Silicon. The techniques outlined in this document enable you to get the most out of your models despite the memory and computational constraints of running them locally.

Remember that each model has its own strengths and weaknesses, so experimentation is key to finding the optimal approach for your specific use case. Use the provided chat interfaces to test different prompting strategies and parameter combinations to achieve the best results.