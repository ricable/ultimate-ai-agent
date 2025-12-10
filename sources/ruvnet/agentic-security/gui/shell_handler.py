from typing import Tuple, Optional
import subprocess
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import shutil

class AiderShellHandler:
    def __init__(self, coder: Coder):
        """Initialize with an existing coder instance"""
        self.coder = coder
        self.io = coder.commands.io
    
    def run_shell_command(self, command: str, share_output: bool = True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Execute a shell command and optionally share output with AI
        Returns: Tuple of (stdout, stderr)
        """
        try:
            # Strip /run prefix if present
            if command.startswith('/run '):
                command = command[5:]
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                env={"PATH": shutil.which("python")}  # Set PATH to find python
            )
            
            # Capture output
            stdout = result.stdout.strip() if result.stdout else None
            stderr = result.stderr.strip() if result.stderr else None
            
            # Share with AI if requested and there's output
            if share_output and (stdout or stderr):
                output = stdout if stdout else stderr
                # Return the output to be handled by GUI's chat system with git-specific handling
                # Provide more contextual guidance based on command type
                if command.startswith('git '):
                    return stdout, stderr, (
                        f"Git command `{command}` output:\n```\n{output}\n```\n\n"
                        "I can help you:\n"
                        "- Explain what these changes mean\n"
                        "- Suggest next git commands\n"
                        "- Help resolve any issues\n"
                        "What would you like to know?"
                    )
                elif command.startswith('python ') or command.endswith('.py'):
                    return stdout, stderr, (
                        f"Python script `{command}` output:\n```\n{output}\n```\n\n"
                        "I can help you:\n"
                        "- Debug any errors\n"
                        "- Explain the output\n"
                        "- Optimize the code\n"
                        "What would you like me to help with?"
                    )
                else:
                    return stdout, stderr, (
                        f"Command `{command}` output:\n```\n{output}\n```\n\n"
                        "I can:\n"
                        "- Explain this output\n"
                        "- Suggest related commands\n"
                        "- Help troubleshoot issues\n"
                        "How can I assist you?"
                    )
            
            return stdout, stderr, None
            
        except Exception as e:
            error_msg = str(e)
            self.io.tool_error(f"Error executing command: {error_msg}")
            return None, error_msg
    
    def run_with_ai_feedback(self, command: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Run command and get AI feedback on the output
        Returns: Tuple of (stdout, stderr, chat_msg)
        """
        stdout, stderr, _ = self.run_shell_command(command, share_output=False)
        
        output = stderr if stderr else stdout
        if output:
            if command.startswith('git '):
                chat_msg = (
                    f"Git command `{command}` output:\n```\n{output}\n```\n\n"
                    "Please provide a detailed analysis including:\n"
                    "1. What changes were made or what state is shown\n"
                    "2. Any potential issues or warnings\n"
                    "3. Recommended next steps\n"
                    "4. Best practices relevant to this operation"
                )
            elif command.startswith('python ') or command.endswith('.py'):
                chat_msg = (
                    f"Python script `{command}` output:\n```\n{output}\n```\n\n"
                    "Please provide a detailed analysis including:\n"
                    "1. Execution results and any errors\n"
                    "2. Code quality insights\n"
                    "3. Performance considerations\n"
                    "4. Security implications if relevant"
                )
            else:
                chat_msg = (
                    f"Command `{command}` output:\n```\n{output}\n```\n\n"
                    "Please provide a detailed analysis including:\n"
                    "1. What the output means\n"
                    "2. Any warnings or issues\n"
                    "3. Relevant system implications\n"
                    "4. Suggested follow-up actions"
                )
            return stdout, stderr, chat_msg
        return stdout, stderr, None
