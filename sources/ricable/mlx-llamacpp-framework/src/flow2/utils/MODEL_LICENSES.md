# Model Licensing Information

This document provides information about the licensing terms for various LLM models that can be downloaded using our model management system. Always check the original license terms before using these models.

## License Types

### Llama 2 Community License

**Models**: Llama 2 family (7B, 13B, 70B)

**Key Terms**:
- Non-commercial use is allowed for all versions
- Commercial use is allowed for models with <700M parameters without additional permission
- Commercial use for models >700M parameters requires approval from Meta
- Requires attribution to Meta AI
- Must include the license text with any distribution
- Cannot use for illegal or harmful purposes
- No warranty provided

**Official License**: [Llama 2 Community License](https://ai.meta.com/llama/license/)

**Registration Required**: Yes, through Hugging Face

### Apache License 2.0

**Models**: Mistral, Falcon, OpenLLaMA

**Key Terms**:
- Free for commercial and non-commercial use
- Requires attribution and copyright notice
- Modifications must be stated
- No liability or warranty
- Patent rights granted to users

**Official License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

**Registration Required**: No

### MIT License

**Models**: Phi-2, TinyLlama

**Key Terms**:
- Free for commercial and non-commercial use
- Very permissive
- Requires only attribution and copyright notice
- No liability or warranty

**Official License**: [MIT License](https://opensource.org/licenses/MIT)

**Registration Required**: No

### Gemma License

**Models**: Gemma 2B, Gemma 7B

**Key Terms**:
- Can be used for commercial and non-commercial purposes
- Prohibits using the models for unlawful purposes
- Prohibits using the models to train models that don't adhere to Google's AI Principles
- No warranty provided

**Official License**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

**Registration Required**: Yes, through Kaggle or Hugging Face

## Model-Specific Licensing Information

### Llama 2 Models

| Model | License | Commercial Use | Registration | Download Access |
|-------|---------|----------------|--------------|----------------|
| Llama 2 7B | Llama 2 Community License | Requires approval | HF account | Gated |
| Llama 2 7B Chat | Llama 2 Community License | Requires approval | HF account | Gated |
| Llama 2 13B | Llama 2 Community License | Requires approval | HF account | Gated |
| Llama 2 13B Chat | Llama 2 Community License | Requires approval | HF account | Gated |

### Mistral Models

| Model | License | Commercial Use | Registration | Download Access |
|-------|---------|----------------|--------------|----------------|
| Mistral 7B | Apache 2.0 | Allowed | None | Open |
| Mistral 7B Instruct | Apache 2.0 | Allowed | None | Open |

### Phi Models

| Model | License | Commercial Use | Registration | Download Access |
|-------|---------|----------------|--------------|----------------|
| Phi-2 | MIT | Allowed | None | Open |

### Gemma Models

| Model | License | Commercial Use | Registration | Download Access |
|-------|---------|----------------|--------------|----------------|
| Gemma 2B | Gemma Terms | Allowed with restrictions | HF account | Gated |
| Gemma 7B | Gemma Terms | Allowed with restrictions | HF account | Gated |

## Accessing Gated Models

Some models require registration and approval before they can be downloaded. To access these models:

1. Create an account on [Hugging Face](https://huggingface.co/)
2. Request access to the specific model repository
3. For Meta's Llama 2, visit [Meta's Llama page](https://ai.meta.com/resources/models-and-libraries/llama/) and accept the terms
4. For Google's Gemma, visit [Google AI Studio](https://ai.google.dev/gemma) and accept the terms
5. Once approved, log in via the Hugging Face CLI:
   ```
   pip install huggingface_hub
   huggingface-cli login
   ```

## Usage Recommendations

When using these models:

1. **Always include proper attribution** as required by the license
2. **Check for commercial use restrictions** before deploying in a commercial product
3. **Keep a copy of the license** with your project or deployment
4. **Monitor for license changes** as terms may be updated over time
5. **Consider data privacy implications** when using these models

## Disclaimer

This information is provided for reference only and may not be complete or up to date. The actual license terms may change, and users should always refer to the original license documentation for each model before use. This document does not constitute legal advice.

## References

- [Hugging Face Model Licenses](https://huggingface.co/docs/hub/models-licenses)
- [Meta AI Llama 2](https://ai.meta.com/llama/)
- [Mistral AI](https://mistral.ai/)
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
- [Google Gemma](https://ai.google.dev/gemma)