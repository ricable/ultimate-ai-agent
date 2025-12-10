# E2B Sandbox Templates

This directory contains sandbox templates used by the SPARC UI for different development environments. Each template provides a pre-configured E2B sandbox environment for specific frameworks and libraries.

## Available Templates

### Gradio Developer
- **Template ID**: `1ypi8ae3amtyxttny60k`
- **Template Name**: `gradio-developer`
- **Purpose**: Build ML/AI interfaces with Gradio
- **Default Port**: 7860
- **Base File**: `app.py`
- **Start Command**: `cd /home/user && gradio app.py`
- **Resources**: 4 CPU cores, 4GB memory

### Next.js Developer
- **Template ID**: `scwxnhs1apt5uj7na7db`
- **Template Name**: `nextjs-developer`
- **Purpose**: Create React applications with Next.js
- **Default Port**: 3000
- **Base File**: `_app.tsx`
- **Start Command**: `/compile_page.sh`
- **Resources**: 4 CPU cores, 4GB memory

### Streamlit Developer
- **Template Name**: `streamlit-developer`
- **Purpose**: Develop data apps with Streamlit
- **Default Port**: 8501
- **Base File**: `app.py`

### Vue Developer
- **Template Name**: `vue-developer`
- **Purpose**: Build Vue/Nuxt applications
- **Default Port**: 3000
- **Base File**: `nuxt.config.ts`

## Template Structure

Each template directory contains:
```
template-name/
├── e2b.Dockerfile    # Docker configuration for the sandbox environment
├── e2b.toml         # E2B configuration defining sandbox setup
└── framework files  # Framework-specific files (e.g., app.py, _app.tsx)
```

## Using Templates

### JavaScript/TypeScript
```typescript
import { Sandbox } from '@e2b/sdk'

// Create a sandbox using template name
const sandbox = await Sandbox.create({
  template: 'gradio-developer'  // or any other template name
})

// Or using template ID
const sandbox = await Sandbox.create({
  template: '1ypi8ae3amtyxttny60k'  // Gradio template ID
})
```

### Python
```python
from e2b import Sandbox

# Create a sandbox using template name
sandbox = Sandbox(template='gradio-developer')

# Or using template ID
sandbox = Sandbox(template='1ypi8ae3amtyxttny60k')
```

## Template Configuration

Each template's `e2b.toml` defines the sandbox configuration:

```toml
template_id = "unique-template-id"
dockerfile = "e2b.Dockerfile"
template_name = "template-name"
start_cmd = "command to start the application"
cpu_count = 4
memory_mb = 4_096
team_id = "460355b3-4f64-48f9-9a16-4442817f79f5"
```

## Integration with SPARC UI

Templates are used through the fragment schema:

```typescript
const fragment = {
  template: "gradio-developer",  // Template name
  port: 7860,                   // Template-specific port
  code: [{
    file_path: "app.py",        // Template-specific file
    file_content: `
      import gradio as gr
      # Your application code
    `
  }]
}
```

## Adding New Templates

1. Create a new directory under `sandbox-templates`:
   ```bash
   mkdir ui/sandbox-templates/[framework]-developer
   ```

2. Create required configuration files:
   - `e2b.Dockerfile`: Define the development environment
   - `e2b.toml`: Configure the E2B sandbox settings
   - Framework-specific files

3. Configure `e2b.toml`:
   ```toml
   template_id = "your-template-id"
   dockerfile = "e2b.Dockerfile"
   template_name = "framework-developer"
   start_cmd = "your-start-command"
   cpu_count = 4
   memory_mb = 4_096
   team_id = "460355b3-4f64-48f9-9a16-4442817f79f5"
   ```

4. Create the Dockerfile:
   ```dockerfile
   FROM node:18-bullseye  # Or appropriate base image
   
   # Install framework-specific dependencies
   RUN npm install -g your-framework-cli
   
   # Set working directory
   WORKDIR /root/workspace
   ```

5. Add any necessary framework configuration files

## Template Best Practices

1. **Naming Convention**:
   - Use lowercase with hyphens
   - Follow the pattern: `[framework]-developer`
   - Example: `python-developer`, `react-developer`

2. **Port Configuration**:
   - Use standard ports when possible
   - Document port usage in `e2b.toml`
   - Handle port conflicts in sandbox creation

3. **Dependencies**:
   - Include common development dependencies in Dockerfile
   - Document additional dependency installation process
   - Test dependency installation workflows

4. **File Structure**:
   - Follow framework conventions
   - Include necessary configuration files
   - Document file structure requirements

## Security Considerations

- All code execution is sandboxed in E2B environments
- Resource limits are enforced (CPU, memory)
- Network access is controlled
- File system access is isolated
