#!/usr/bin/env python3
"""
MCP Tool Training Dataset Generator
Creates high-quality training data for Claude Agent SDK and MCP tools
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCPToolExample:
    """Single MCP tool training example"""
    user_query: str
    tool_name: str
    tool_params: Dict[str, Any]
    expected_response: str
    context: str = ""
    difficulty: str = "medium"  # easy, medium, hard


class MCPDatasetBuilder:
    """Build comprehensive MCP tool calling dataset"""

    def __init__(self):
        self.examples: List[MCPToolExample] = []

    def add_read_examples(self):
        """Add Read tool examples"""
        examples = [
            MCPToolExample(
                user_query="Read the package.json file",
                tool_name="read",
                tool_params={"file_path": "package.json"},
                expected_response="I'll read the package.json file.\n<tool_use>read(file_path='package.json')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Show me the contents of src/index.ts starting from line 50",
                tool_name="read",
                tool_params={"file_path": "src/index.ts", "offset": 50, "limit": 50},
                expected_response="I'll read src/index.ts starting from line 50.\n<tool_use>read(file_path='src/index.ts', offset=50, limit=50)</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="What's in the authentication configuration file?",
                tool_name="read",
                tool_params={"file_path": "config/auth.config.js"},
                expected_response="I'll read the authentication configuration.\n<tool_use>read(file_path='config/auth.config.js')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Read the README and tell me about installation",
                tool_name="read",
                tool_params={"file_path": "README.md"},
                expected_response="I'll read the README file to find installation instructions.\n<tool_use>read(file_path='README.md')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_write_examples(self):
        """Add Write tool examples"""
        examples = [
            MCPToolExample(
                user_query="Create a new file called utils.ts with a helper function",
                tool_name="write",
                tool_params={"file_path": "src/utils.ts", "content": "export function formatDate(date: Date): string {\n  return date.toISOString();\n}"},
                expected_response="I'll create utils.ts with a date formatting helper.\n<tool_use>write(file_path='src/utils.ts', content='export function formatDate(date: Date): string {\\n  return date.toISOString();\\n}')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Write a basic Express server to server.js",
                tool_name="write",
                tool_params={"file_path": "server.js", "content": "const express = require('express');\nconst app = express();\n\napp.get('/', (req, res) => {\n  res.send('Hello World!');\n});\n\napp.listen(3000);"},
                expected_response="I'll create a basic Express server.\n<tool_use>write(file_path='server.js', content='const express = require(\\'express\\');\\nconst app = express();\\n\\napp.get(\\'/\\', (req, res) => {\\n  res.send(\\'Hello World!\\');\\n});\\n\\napp.listen(3000);')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Create a .gitignore file",
                tool_name="write",
                tool_params={"file_path": ".gitignore", "content": "node_modules/\n.env\ndist/\n*.log"},
                expected_response="I'll create a .gitignore file with common exclusions.\n<tool_use>write(file_path='.gitignore', content='node_modules/\\n.env\\ndist/\\n*.log')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_edit_examples(self):
        """Add Edit tool examples"""
        examples = [
            MCPToolExample(
                user_query="Change the port from 3000 to 8080 in server.js",
                tool_name="edit",
                tool_params={"file_path": "server.js", "old_string": "app.listen(3000)", "new_string": "app.listen(8080)"},
                expected_response="I'll update the port to 8080.\n<tool_use>edit(file_path='server.js', old_string='app.listen(3000)', new_string='app.listen(8080)')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Update the function name from getData to fetchData",
                tool_name="edit",
                tool_params={"file_path": "src/api.ts", "old_string": "function getData", "new_string": "function fetchData"},
                expected_response="I'll rename the function to fetchData.\n<tool_use>edit(file_path='src/api.ts', old_string='function getData', new_string='function fetchData')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Fix the typo in the error message - change 'occured' to 'occurred'",
                tool_name="edit",
                tool_params={"file_path": "src/errors.ts", "old_string": "Error occured", "new_string": "Error occurred"},
                expected_response="I'll fix the typo in the error message.\n<tool_use>edit(file_path='src/errors.ts', old_string='Error occured', new_string='Error occurred')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_bash_examples(self):
        """Add Bash tool examples"""
        examples = [
            MCPToolExample(
                user_query="Install the dependencies",
                tool_name="bash",
                tool_params={"command": "npm install", "description": "Install npm dependencies"},
                expected_response="I'll install the dependencies using npm.\n<tool_use>bash(command='npm install', description='Install npm dependencies')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Run the test suite",
                tool_name="bash",
                tool_params={"command": "npm test", "description": "Run test suite"},
                expected_response="I'll run the test suite.\n<tool_use>bash(command='npm test', description='Run test suite')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Check the git status",
                tool_name="bash",
                tool_params={"command": "git status", "description": "Check git repository status"},
                expected_response="I'll check the git status.\n<tool_use>bash(command='git status', description='Check git repository status')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Build the project and run it",
                tool_name="bash",
                tool_params={"command": "npm run build && npm start", "description": "Build and start the application"},
                expected_response="I'll build the project and start it.\n<tool_use>bash(command='npm run build && npm start', description='Build and start the application')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Create a production build",
                tool_name="bash",
                tool_params={"command": "npm run build", "description": "Create production build"},
                expected_response="I'll create a production build.\n<tool_use>bash(command='npm run build', description='Create production build')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_grep_examples(self):
        """Add Grep tool examples"""
        examples = [
            MCPToolExample(
                user_query="Search for all TODO comments in the code",
                tool_name="grep",
                tool_params={"pattern": "TODO", "output_mode": "content", "glob": "**/*.ts"},
                expected_response="I'll search for TODO comments across TypeScript files.\n<tool_use>grep(pattern='TODO', output_mode='content', glob='**/*.ts')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Find all authentication-related functions",
                tool_name="grep",
                tool_params={"pattern": "function.*auth", "output_mode": "files_with_matches"},
                expected_response="I'll search for authentication functions.\n<tool_use>grep(pattern='function.*auth', output_mode='files_with_matches')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Look for API endpoint definitions",
                tool_name="grep",
                tool_params={"pattern": "app\\.(get|post|put|delete)", "output_mode": "content"},
                expected_response="I'll search for API endpoint definitions.\n<tool_use>grep(pattern='app\\\\.(get|post|put|delete)', output_mode='content')</tool_use>",
                difficulty="hard"
            ),
            MCPToolExample(
                user_query="Find where the UserService class is used",
                tool_name="grep",
                tool_params={"pattern": "UserService", "output_mode": "files_with_matches"},
                expected_response="I'll find files that use UserService.\n<tool_use>grep(pattern='UserService', output_mode='files_with_matches')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_glob_examples(self):
        """Add Glob tool examples"""
        examples = [
            MCPToolExample(
                user_query="List all TypeScript files in the src directory",
                tool_name="glob",
                tool_params={"pattern": "src/**/*.ts"},
                expected_response="I'll list all TypeScript files in src/.\n<tool_use>glob(pattern='src/**/*.ts')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="Show me all test files",
                tool_name="glob",
                tool_params={"pattern": "**/*.test.ts"},
                expected_response="I'll find all test files.\n<tool_use>glob(pattern='**/*.test.ts')</tool_use>",
                difficulty="easy"
            ),
            MCPToolExample(
                user_query="List all configuration files",
                tool_name="glob",
                tool_params={"pattern": "**/*.config.{js,ts,json}"},
                expected_response="I'll list all configuration files.\n<tool_use>glob(pattern='**/*.config.{js,ts,json}')</tool_use>",
                difficulty="medium"
            ),
            MCPToolExample(
                user_query="Find all JavaScript files except node_modules",
                tool_name="glob",
                tool_params={"pattern": "**/*.js"},
                expected_response="I'll find all JavaScript files.\n<tool_use>glob(pattern='**/*.js')</tool_use>",
                difficulty="easy"
            ),
        ]
        self.examples.extend(examples)

    def add_agentic_workflow_examples(self):
        """Add multi-step agentic workflow examples"""
        examples = [
            MCPToolExample(
                user_query="Create a new API endpoint for user registration",
                tool_name="multi_step",
                tool_params={},
                expected_response="I'll create a user registration endpoint. Let me break this down:\n1. Create the route handler\n2. Add validation logic\n3. Write tests\n\nLet's start:\n<tool_use>write(file_path='src/routes/auth.ts', content='import { Router } from \\'express\\';\\n\\nconst router = Router();\\n\\nrouter.post(\\'/register\\', async (req, res) => {\\n  // Implementation\\n});\\n\\nexport default router;')</tool_use>",
                difficulty="hard",
                context="Multi-step workflow requiring planning"
            ),
            MCPToolExample(
                user_query="Debug why the authentication is failing",
                tool_name="multi_step",
                tool_params={},
                expected_response="I'll help debug the authentication issue. Let me investigate:\n1. Check the auth configuration\n2. Review recent changes\n3. Look at error logs\n\nStarting with the configuration:\n<tool_use>read(file_path='config/auth.config.js')</tool_use>",
                difficulty="hard",
                context="Debugging workflow"
            ),
            MCPToolExample(
                user_query="Refactor the user service to use async/await",
                tool_name="multi_step",
                tool_params={},
                expected_response="I'll refactor the user service to use async/await. My approach:\n1. Read the current implementation\n2. Identify callback patterns\n3. Convert to async/await\n4. Update tests\n\nLet me start by reading the file:\n<tool_use>read(file_path='src/services/userService.js')</tool_use>",
                difficulty="hard",
                context="Refactoring workflow"
            ),
        ]
        self.examples.extend(examples)

    def add_claude_flow_examples(self):
        """Add Claude Flow specific examples"""
        examples = [
            MCPToolExample(
                user_query="Initialize a swarm for code review",
                tool_name="bash",
                tool_params={"command": "npx claude-flow swarm init --topology mesh --agents 5", "description": "Initialize code review swarm"},
                expected_response="I'll initialize a mesh swarm with 5 agents for code review.\n<tool_use>bash(command='npx claude-flow swarm init --topology mesh --agents 5', description='Initialize code review swarm')</tool_use>",
                difficulty="medium",
                context="Claude Flow swarm coordination"
            ),
            MCPToolExample(
                user_query="Run the Flow Nexus neural network trainer",
                tool_name="bash",
                tool_params={"command": "npx flow-nexus neural train --model phi4 --epochs 10", "description": "Train neural network"},
                expected_response="I'll start neural network training with Flow Nexus.\n<tool_use>bash(command='npx flow-nexus neural train --model phi4 --epochs 10', description='Train neural network')</tool_use>",
                difficulty="medium",
                context="Flow Nexus integration"
            ),
            MCPToolExample(
                user_query="Deploy a sandbox for testing",
                tool_name="bash",
                tool_params={"command": "npx flow-nexus sandbox create --template node --name test-env", "description": "Create test sandbox"},
                expected_response="I'll create a Node.js sandbox for testing.\n<tool_use>bash(command='npx flow-nexus sandbox create --template node --name test-env', description='Create test sandbox')</tool_use>",
                difficulty="medium",
                context="Flow Nexus sandbox"
            ),
        ]
        self.examples.extend(examples)

    def build_dataset(self) -> List[Dict[str, str]]:
        """Build complete training dataset"""
        logger.info("Building MCP tool training dataset...")

        # Add all example categories
        self.add_read_examples()
        self.add_write_examples()
        self.add_edit_examples()
        self.add_bash_examples()
        self.add_grep_examples()
        self.add_glob_examples()
        self.add_agentic_workflow_examples()
        self.add_claude_flow_examples()

        # Convert to training format
        dataset = []
        for example in self.examples:
            # Format as conversation
            conversation = f"User: {example.user_query}\nAssistant: {example.expected_response}"

            dataset.append({
                "text": conversation,
                "tool_name": example.tool_name,
                "difficulty": example.difficulty,
                "context": example.context
            })

        logger.info(f"Dataset created with {len(dataset)} examples")
        logger.info(f"  - Read: {sum(1 for d in dataset if d['tool_name'] == 'read')}")
        logger.info(f"  - Write: {sum(1 for d in dataset if d['tool_name'] == 'write')}")
        logger.info(f"  - Edit: {sum(1 for d in dataset if d['tool_name'] == 'edit')}")
        logger.info(f"  - Bash: {sum(1 for d in dataset if d['tool_name'] == 'bash')}")
        logger.info(f"  - Grep: {sum(1 for d in dataset if d['tool_name'] == 'grep')}")
        logger.info(f"  - Glob: {sum(1 for d in dataset if d['tool_name'] == 'glob')}")
        logger.info(f"  - Multi-step: {sum(1 for d in dataset if d['tool_name'] == 'multi_step')}")

        return dataset

    def save_dataset(self, output_path: str):
        """Save dataset to file"""
        dataset = self.build_dataset()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Dataset saved to {output_path}")

    def get_validation_examples(self) -> List[Dict[str, str]]:
        """Get examples specifically for validation"""
        validation_examples = [
            {
                "query": "Read the tsconfig.json file",
                "expected_tool": "read",
                "expected_params": {"file_path": "tsconfig.json"}
            },
            {
                "query": "Create a new component file",
                "expected_tool": "write",
                "expected_params": {"file_path": "src/components/NewComponent.tsx"}
            },
            {
                "query": "Run the linter",
                "expected_tool": "bash",
                "expected_params": {"command": "npm run lint"}
            },
            {
                "query": "Find all error handling code",
                "expected_tool": "grep",
                "expected_params": {"pattern": "catch|throw"}
            },
            {
                "query": "List all React components",
                "expected_tool": "glob",
                "expected_params": {"pattern": "src/components/**/*.tsx"}
            },
        ]

        return validation_examples


def main():
    """Generate and save MCP training dataset"""
    builder = MCPDatasetBuilder()

    # Build and save dataset
    output_path = "/app/training/data/mcp_tools_dataset.json"
    builder.save_dataset(output_path)

    # Save validation examples
    validation_examples = builder.get_validation_examples()
    validation_path = "/app/validation/data/mcp_validation_set.json"

    Path(validation_path).parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, 'w') as f:
        json.dump(validation_examples, f, indent=2)

    logger.info(f"Validation set saved to {validation_path}")
    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()
