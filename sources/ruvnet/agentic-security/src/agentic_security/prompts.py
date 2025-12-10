import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    DEFAULT_PROMPTS = {
        'architecture_review': """Review the architecture for security vulnerabilities and suggest improvements:
1. Identify potential security weaknesses
2. Suggest architectural improvements
3. Recommend security best practices
Analysis:""",
        
        'fix_generation': """Generate secure fixes for the following vulnerability:
{vulnerability_type} in file {file_path}

Required Changes:
1. Replace any insecure functions/methods with their secure alternatives
2. Add proper input validation and sanitization
3. Use safe defaults and secure configuration options

Security Considerations:
1. Follow the principle of least privilege
2. Implement defense in depth
3. Use well-tested security libraries
4. Add appropriate error handling

Code Guidelines:
1. Make minimal necessary changes
2. Maintain existing code structure
3. Add clear security-focused comments
4. Ensure backward compatibility

Please provide the exact code changes needed, using secure alternatives like:
- subprocess.run() instead of os.system()
- parameterized queries instead of string formatting
- defusedxml instead of xml.etree
- bcrypt/argon2 instead of md5/sha1
- html.escape() for XSS prevention

Proposed fix:""",
        
        'pr_description': """Create a detailed pull request description for these security changes:
{changes}

Please include:
1. Summary of Security Changes
   - List each vulnerability fixed
   - Impact of each fix
   - Files modified

2. Security Impact Analysis
   - Risk level before/after changes
   - Attack vectors addressed
   - Potential side effects

3. Testing & Validation
   - Security tests performed
   - Manual validation steps
   - Regression testing results

4. Implementation Notes
   - Security best practices applied
   - Dependencies updated
   - Configuration changes

Description:"""
    }

    def __init__(self, custom_prompts: Optional[Dict[str, str]] = None):
        self.prompts = self.DEFAULT_PROMPTS.copy()
        if custom_prompts:
            self.prompts.update(custom_prompts)

    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input strings for AI prompts
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            Sanitized string safe for AI prompts
        """
        if not isinstance(input_str, str):
            input_str = str(input_str)
            
        # Remove any potentially problematic characters
        sanitized = input_str.replace('"', '\\"')  # Escape quotes
        sanitized = sanitized.replace('$', '\\$')  # Escape dollar signs
        sanitized = sanitized.replace('`', '\\`')  # Escape backticks
        
        # Limit length
        max_length = 8000  # Reasonable limit for most AI models
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
            
        return sanitized
        
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get formatted prompt"""
        if prompt_type not in self.prompts:
            logger.error(f"Unknown prompt type requested: {prompt_type}")
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        prompt = self.prompts[prompt_type].format(**kwargs)
        logger.info(f"Generated {prompt_type} prompt:")
        logger.info(prompt)
        return prompt
