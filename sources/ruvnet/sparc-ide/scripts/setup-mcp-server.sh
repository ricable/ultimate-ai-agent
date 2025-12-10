#!/bin/bash
# SPARC IDE - MCP Server Setup Script
# This script sets up the Model Context Protocol (MCP) server for SPARC IDE

set -e
set -o nounset  # Exit if a variable is unset
set -o pipefail # Exit if any command in a pipeline fails

# Configuration
MCP_DIR="src/mcp"
MCP_SERVER_PORT=3001
NODE_VERSION="20"

# Print colored output
print_info() {
    echo -e "\e[1;34m[INFO]\e[0m $1"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS]\e[0m $1"
}

print_error() {
    echo -e "\e[1;31m[ERROR]\e[0m $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js $NODE_VERSION or later and try again."
        exit 1
    fi
    
    # Check Node.js version
    NODE_CURRENT_VERSION=$(node -v | cut -d 'v' -f 2)
    if [ "$(printf '%s\n' "$NODE_VERSION" "$NODE_CURRENT_VERSION" | sort -V | head -n1)" != "$NODE_VERSION" ]; then
        print_error "Node.js version $NODE_CURRENT_VERSION is less than required version $NODE_VERSION. Please upgrade Node.js and try again."
        exit 1
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
    
    print_success "All prerequisites are met."
}

# Create MCP server directory structure
create_directory_structure() {
    print_info "Creating MCP server directory structure..."
    
    mkdir -p "$MCP_DIR/src"
    mkdir -p "$MCP_DIR/config"
    mkdir -p "$MCP_DIR/tools"
    mkdir -p "$MCP_DIR/resources"
    
    print_success "MCP server directory structure created."
}

# Create MCP server configuration
create_configuration() {
    print_info "Creating MCP server configuration..."
    
    # Create package.json
    cat > "$MCP_DIR/package.json" << 'EOL'
{
  "name": "sparc-mcp-server",
  "version": "1.0.0",
  "description": "Model Context Protocol (MCP) server for SPARC IDE",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "test": "jest",
    "generate-certs": "mkdir -p ./certs && openssl req -x509 -newkey rsa:4096 -keyout ./certs/key.pem -out ./certs/cert.pem -days 365 -nodes",
    "generate-secret": "node -e \"console.log(require('crypto').randomBytes(32).toString('hex'))\""
  },
  "dependencies": {
    "express": "4.18.2",
    "cors": "2.8.5",
    "body-parser": "1.20.2",
    "dotenv": "16.3.1",
    "winston": "3.11.0",
    "axios": "1.6.2",
    "helmet": "7.1.0",
    "express-rate-limit": "7.1.5",
    "jsonwebtoken": "9.0.2",
    "bcrypt": "5.1.1",
    "crypto": "1.0.1"
  },
  "devDependencies": {
    "nodemon": "3.0.2",
    "jest": "29.7.0",
    "supertest": "6.3.3"
  },
  "engines": {
    "node": ">=20.0.0"
  }
}
EOL
    
    # Create .env.example file (template for users to create their own .env file)
    cat > "$MCP_DIR/.env.example" << 'EOL'
# MCP Server Configuration
MCP_SERVER_PORT=3001
MCP_SERVER_HOST=localhost

# API Keys - DO NOT HARDCODE ACTUAL KEYS HERE
# Use environment variables or a secure credential manager
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=

# Tool Configuration
ENABLE_CODE_ANALYSIS=true
ENABLE_CODE_MODIFICATION=true
ENABLE_CODE_SEARCH=true

# Security Configuration
ENABLE_HTTPS=true
HTTPS_KEY_PATH=./certs/key.pem
HTTPS_CERT_PATH=./certs/cert.pem
ENABLE_AUTH=true
AUTH_SECRET=

# Admin Authentication
# Generate a secure password hash using ./scripts/generate-admin-password.sh
MCP_ADMIN_USERNAME=admin
MCP_ADMIN_PASSWORD_HASH=
EOL

    # Create a README file with instructions for secure credential management
    cat > "$MCP_DIR/README.md" << 'EOL'
# SPARC MCP Server

## Secure Credential Management

For security reasons, API keys and secrets should never be hardcoded in your application files.
Instead, follow these secure practices:

1. Copy `.env.example` to `.env` (this file should never be committed to version control)
2. Set your API keys using one of these secure methods:
   - Use environment variables
   - Use a secure credential manager like HashiCorp Vault or AWS Secrets Manager
   - For development only: Add them to your `.env` file, but ensure this file is in your `.gitignore`

3. For HTTPS setup:
   - Generate self-signed certificates for development:
     ```
     mkdir -p ./certs
     openssl req -x509 -newkey rsa:4096 -keyout ./certs/key.pem -out ./certs/cert.pem -days 365 -nodes
     ```
   - For production, use properly signed certificates from a trusted CA

4. For authentication:
   - Generate a strong random secret:
     ```
     node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
     ```
   - Add this secret to your `.env` file as AUTH_SECRET

## Never commit sensitive information to version control!
# Create .env file with secure defaults
cat > "$MCP_DIR/.env" << 'EOL'
# MCP Server Configuration
MCP_SERVER_PORT=3001
MCP_SERVER_HOST=localhost

# API Keys - DO NOT HARDCODE ACTUAL KEYS HERE
# Use environment variables or a secure credential manager
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=

# Tool Configuration
ENABLE_CODE_ANALYSIS=true
ENABLE_CODE_MODIFICATION=true
ENABLE_CODE_SEARCH=true

# Security Configuration
ENABLE_HTTPS=true
HTTPS_KEY_PATH=./certs/key.pem
HTTPS_CERT_PATH=./certs/cert.pem
ENABLE_AUTH=true
AUTH_SECRET=

# Admin Authentication
# Generate a secure password hash using ./scripts/generate-admin-password.sh
MCP_ADMIN_USERNAME=admin
MCP_ADMIN_PASSWORD_HASH=
EOL

    # Create .gitignore file to prevent committing sensitive files
    cat > "$MCP_DIR/.gitignore" << 'EOL'
# Environment variables
.env

# Certificates
/certs

# Logs
*.log
logs/

# Dependencies
/node_modules

# Build artifacts
/dist
/build
EOL
    
    # Create MCP server configuration
    cat > "$MCP_DIR/config/server.js" << 'EOL'
// MCP Server Configuration

require('dotenv').config();
const fs = require('fs');
const path = require('path');

/**
 * Securely load API keys from environment variables
 * @param {string} keyName - Name of the API key
 * @returns {string|null} - The API key or null if not found
 */
function getSecureApiKey(keyName) {
  const key = process.env[keyName];
  if (!key) {
    console.warn(`Warning: ${keyName} not found in environment variables`);
    return null;
  }
  return key;
}

/**
 * Validate that a required configuration value exists
 * @param {string} value - The value to check
 * @param {string} name - Name of the configuration value
 * @param {string} defaultValue - Default value to use if not provided
 * @returns {string} - The value or default
 */
function validateConfig(value, name, defaultValue) {
  if (!value && !defaultValue) {
    throw new Error(`Required configuration ${name} is missing and has no default`);
  }
  return value || defaultValue;
}

// HTTPS configuration
const httpsEnabled = process.env.ENABLE_HTTPS === 'true';
let httpsOptions = null;

if (httpsEnabled) {
  try {
    const keyPath = validateConfig(process.env.HTTPS_KEY_PATH, 'HTTPS_KEY_PATH', './certs/key.pem');
    const certPath = validateConfig(process.env.HTTPS_CERT_PATH, 'HTTPS_CERT_PATH', './certs/cert.pem');
    
    httpsOptions = {
      key: fs.readFileSync(path.resolve(keyPath)),
      cert: fs.readFileSync(path.resolve(certPath))
    };
  } catch (error) {
    console.error('Error loading HTTPS certificates:', error.message);
    console.error('Falling back to HTTP (not secure for production)');
    httpsOptions = null;
  }
}

// Authentication configuration
const authEnabled = process.env.ENABLE_AUTH === 'true';
const authSecret = process.env.AUTH_SECRET;

if (authEnabled && !authSecret) {
  console.error('Warning: Authentication is enabled but AUTH_SECRET is not set');
  console.error('This is a security risk. Generate a strong secret and set it in your .env file');
}

module.exports = {
  server: {
    port: validateConfig(process.env.MCP_SERVER_PORT, 'MCP_SERVER_PORT', '3001'),
    host: validateConfig(process.env.MCP_SERVER_HOST, 'MCP_SERVER_HOST', 'localhost'),
    https: {
      enabled: httpsEnabled,
      options: httpsOptions
    }
  },
  auth: {
    enabled: authEnabled,
    secret: authSecret,
    tokenExpiration: '24h'
  },
  apiKeys: {
    openai: getSecureApiKey('OPENAI_API_KEY'),
    anthropic: getSecureApiKey('ANTHROPIC_API_KEY'),
    openrouter: getSecureApiKey('OPENROUTER_API_KEY'),
    google: getSecureApiKey('GOOGLE_API_KEY')
  },
  tools: {
    enableCodeAnalysis: process.env.ENABLE_CODE_ANALYSIS === 'true',
    enableCodeModification: process.env.ENABLE_CODE_MODIFICATION === 'true',
    enableCodeSearch: process.env.ENABLE_CODE_SEARCH === 'true'
  },
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    file: process.env.LOG_FILE || 'mcp-server.log'
  },
  security: {
    rateLimiting: {
      enabled: true,
      maxRequests: 100,
      timeWindow: 60 * 1000 // 1 minute
    },
    cors: {
      // Default to localhost only for security
      // In production, set CORS_ORIGIN to your specific domain
      origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
      methods: ['GET', 'POST'],
      credentials: true,
      allowedHeaders: ['Content-Type', 'Authorization'],
      maxAge: 86400 // 24 hours
    }
  }
};
EOL
    
    print_success "MCP server configuration created."
}

# Create MCP server main file
create_main_file() {
    print_info "Creating MCP server main file..."
    
    cat > "$MCP_DIR/src/index.js" << 'EOL'
// SPARC IDE - MCP Server
// This is the main entry point for the MCP server

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const winston = require('winston');
const fs = require('fs');
const https = require('https');
const http = require('http');
const path = require('path');
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const config = require('../config/server');

// Initialize logger
const logger = winston.createLogger({
  level: config.logging.level,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: config.logging.file })
  ]
});

// Initialize Express app
const app = express();

// Security middleware
app.use(helmet()); // Add security headers

// Rate limiting
if (config.security.rateLimiting.enabled) {
  const limiter = rateLimit({
    windowMs: config.security.rateLimiting.timeWindow,
    max: config.security.rateLimiting.maxRequests,
    message: { error: 'Too many requests, please try again later' }
  });
  app.use(limiter);
}

// CORS configuration with enhanced security
app.use(cors({
  origin: config.security.cors.origin,
  methods: config.security.cors.methods,
  credentials: config.security.cors.credentials,
  allowedHeaders: config.security.cors.allowedHeaders,
  maxAge: config.security.cors.maxAge,
  // Add security headers
  preflightContinue: false,
  optionsSuccessStatus: 204
}));

app.use(bodyParser.json());

// Request logging middleware
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.url}`);
  next();
});

// Authentication middleware
const authenticateToken = (req, res, next) => {
  // Skip authentication if disabled
  if (!config.auth.enabled) {
    return next();
  }

  // Get auth header
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  try {
    const user = jwt.verify(token, config.auth.secret);
    req.user = user;
    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    return res.status(403).json({ error: 'Invalid or expired token' });
  }
};

// Load tools
const analyzeCode = require('../tools/analyze-code');
const modifyCode = require('../tools/modify-code');
const searchCode = require('../tools/search-code');

// Define routes
app.get('/', (req, res) => {
  res.json({
    name: 'SPARC IDE MCP Server',
    version: '1.0.0',
    status: 'running',
    https: config.server.https.enabled ? 'enabled' : 'disabled',
    auth: config.auth.enabled ? 'enabled' : 'disabled'
  });
});

// Authentication routes
app.post('/auth/login', (req, res) => {
  // This is a placeholder for actual authentication
  // In a real implementation, you would validate credentials against a database
  if (!config.auth.enabled) {
    return res.status(404).json({ error: 'Authentication is disabled' });
  }

  const { username, password } = req.body;
  
  // Validate input
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }
  
  // Prevent timing attacks by using constant-time comparison
  const bcrypt = require('bcrypt');
  
  // In a real implementation, you would load these from a secure database
  // For now, we'll use environment variables with a secure default
  const ADMIN_USERNAME = process.env.MCP_ADMIN_USERNAME || 'admin';
  const ADMIN_PASSWORD_HASH = process.env.MCP_ADMIN_PASSWORD_HASH;
  
  // If no password hash is set, reject all login attempts for security
  if (!ADMIN_PASSWORD_HASH) {
    logger.error('No admin password hash set. Set MCP_ADMIN_PASSWORD_HASH in .env file');
    return res.status(500).json({ error: 'Server authentication not configured properly' });
  }
  
  // Check username using constant-time comparison to prevent timing attacks
  const usernameMatches = crypto.timingSafeEqual(
    Buffer.from(username),
    Buffer.from(ADMIN_USERNAME)
  );
  
  if (usernameMatches) {
    // Verify password hash
    bcrypt.compare(password, ADMIN_PASSWORD_HASH, (err, result) => {
      if (err) {
        logger.error('Error verifying password:', err);
        return res.status(500).json({ error: 'Authentication error' });
      }
      
      if (result) {
        // Generate JWT token
        const token = jwt.sign(
          { username, role: 'admin' },
          config.auth.secret,
          { expiresIn: config.auth.tokenExpiration }
        );
        
        // Log successful login (but don't log credentials)
        logger.info(`Successful login for user: ${username}`);
        
        return res.json({ token });
      } else {
        // Log failed attempt (but don't log credentials)
        logger.warn(`Failed login attempt for user: ${username}`);
        return res.status(401).json({ error: 'Invalid credentials' });
      }
    });
  } else {
    // Log failed attempt (but don't log credentials)
    logger.warn(`Failed login attempt for user: ${username}`);
    return res.status(401).json({ error: 'Invalid credentials' });
  }
});

// MCP manifest endpoint
app.get('/mcp/manifest', (req, res) => {
  const manifest = {
    name: 'sparc2-mcp',
    display_name: 'SPARC IDE MCP Server',
    description: 'Model Context Protocol server for SPARC IDE',
    version: '1.0.0',
    tools: [],
    resources: [],
    auth_required: config.auth.enabled
  };
  
  // Add tools based on configuration
  if (config.tools.enableCodeAnalysis) {
    manifest.tools.push({
      name: 'analyze_code',
      description: 'Analyze code to provide insights and suggestions',
      input_schema: {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'The code to analyze'
          },
          language: {
            type: 'string',
            description: 'The programming language of the code'
          }
        },
        required: ['code']
      }
    });
  }
  
  if (config.tools.enableCodeModification) {
    manifest.tools.push({
      name: 'modify_code',
      description: 'Modify code based on instructions',
      input_schema: {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'The code to modify'
          },
          instructions: {
            type: 'string',
            description: 'Instructions for modifying the code'
          },
          language: {
            type: 'string',
            description: 'The programming language of the code'
          }
        },
        required: ['code', 'instructions']
      }
    });
  }
  
  if (config.tools.enableCodeSearch) {
    manifest.tools.push({
      name: 'search_code',
      description: 'Search code for patterns or specific content',
      input_schema: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'The pattern to search for'
          },
          files: {
            type: 'array',
            items: {
              type: 'string'
            },
            description: 'The files to search in'
          }
        },
        required: ['pattern']
      }
    });
  }
  
  // Add resources
  manifest.resources = [
    {
      name: 'code_files',
      description: 'Access to code files in the workspace',
      uri_patterns: ['src://**']
    }
  ];
  
  res.json(manifest);
});

// Secure all tool endpoints with authentication
app.use('/mcp/tools', authenticateToken);

// Tool endpoints
app.post('/mcp/tools/analyze_code', async (req, res) => {
  try {
    if (!config.tools.enableCodeAnalysis) {
      return res.status(404).json({ error: 'Tool not enabled' });
    }
    
    const { code, language } = req.body;
    
    // Input validation
    if (!code || typeof code !== 'string') {
      return res.status(400).json({ error: 'Invalid code parameter' });
    }
    
    const result = await analyzeCode(code, language);
    res.json(result);
  } catch (error) {
    logger.error('Error analyzing code:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/mcp/tools/modify_code', async (req, res) => {
  try {
    if (!config.tools.enableCodeModification) {
      return res.status(404).json({ error: 'Tool not enabled' });
    }
    
    const { code, instructions, language } = req.body;
    
    // Input validation
    if (!code || typeof code !== 'string') {
      return res.status(400).json({ error: 'Invalid code parameter' });
    }
    
    if (!instructions || typeof instructions !== 'string') {
      return res.status(400).json({ error: 'Invalid instructions parameter' });
    }
    
    const result = await modifyCode(code, instructions, language);
    res.json(result);
  } catch (error) {
    logger.error('Error modifying code:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/mcp/tools/search_code', async (req, res) => {
  try {
    if (!config.tools.enableCodeSearch) {
      return res.status(404).json({ error: 'Tool not enabled' });
    }
    
    const { pattern, files } = req.body;
    
    // Input validation
    if (!pattern || typeof pattern !== 'string') {
      return res.status(400).json({ error: 'Invalid pattern parameter' });
    }
    
    // Validate files array if provided
    if (files && !Array.isArray(files)) {
      return res.status(400).json({ error: 'Files parameter must be an array' });
    }
    
    const result = await searchCode(pattern, files);
    res.json(result);
  } catch (error) {
    logger.error('Error searching code:', error);
    res.status(500).json({ error: error.message });
  }
});

// Secure resource endpoints with authentication
app.use('/mcp/resources', authenticateToken);

// Resource endpoints
app.get('/mcp/resources/code_files/*', (req, res) => {
  // Implementation for accessing code files
  // This is a placeholder and would need to be implemented based on your specific needs
  res.status(501).json({ error: 'Not implemented yet' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const PORT = config.server.port;
const HOST = config.server.host;

// Create server based on HTTPS configuration
let server;

if (config.server.https.enabled && config.server.https.options) {
  // HTTPS server
  server = https.createServer(config.server.https.options, app);
  server.listen(PORT, HOST, () => {
    logger.info(`MCP server running at https://${HOST}:${PORT}`);
  });
} else {
  // HTTP server (fallback)
  server = http.createServer(app);
  server.listen(PORT, HOST, () => {
    logger.info(`MCP server running at http://${HOST}:${PORT}`);
    if (process.env.NODE_ENV === 'production') {
      logger.warn('Running in production without HTTPS is not recommended');
    }
  });
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});
EOL
    
    print_success "MCP server main file created."
}

# Create MCP server tool files
create_tool_files() {
    print_info "Creating MCP server tool files..."
    
    # Create analyze-code.js
    cat > "$MCP_DIR/tools/analyze-code.js" << 'EOL'
// Code Analysis Tool

const axios = require('axios');
const config = require('../config/server');

/**
 * Analyzes code and provides insights and suggestions
 * @param {string} code - The code to analyze
 * @param {string} language - The programming language of the code
 * @returns {Promise<object>} - Analysis results
 */
async function analyzeCode(code, language = 'javascript') {
  // This is a placeholder implementation
  // In a real implementation, you might use a language server, static analysis tool,
  // or AI service to analyze the code
  
  // Simple analysis based on code length and complexity
  const lines = code.split('\n');
  const lineCount = lines.length;
  const characterCount = code.length;
  const averageLineLength = characterCount / lineCount;
  
  // Simple complexity analysis (very basic)
  const complexityIndicators = {
    conditionals: (code.match(/if|else|switch|case|?:/g) || []).length,
    loops: (code.match(/for|while|do/g) || []).length,
    functions: (code.match(/function|=>/g) || []).length,
    classes: (code.match(/class/g) || []).length
  };
  
  // Calculate a simple complexity score
  const complexityScore = 
    complexityIndicators.conditionals * 2 + 
    complexityIndicators.loops * 3 + 
    complexityIndicators.functions + 
    complexityIndicators.classes * 4;
  
  // Generate suggestions based on simple heuristics
  const suggestions = [];
  
  if (averageLineLength > 80) {
    suggestions.push({
      type: 'style',
      message: 'Consider breaking long lines into smaller ones for better readability',
      severity: 'info'
    });
  }
  
  if (complexityScore > 50) {
    suggestions.push({
      type: 'complexity',
      message: 'Code appears to be complex. Consider refactoring into smaller functions or methods',
      severity: 'warning'
    });
  }
  
  if (lineCount > 200) {
    suggestions.push({
      type: 'organization',
      message: 'File is quite large. Consider splitting into multiple files',
      severity: 'info'
    });
  }
  
  // Check for common issues based on language
  if (language === 'javascript' || language === 'typescript') {
    if (code.includes('var ')) {
      suggestions.push({
        type: 'modernization',
        message: 'Consider using let or const instead of var for better scoping',
        severity: 'info'
      });
    }
    
    if (code.includes('new Array(') || code.includes('new Object(')) {
      suggestions.push({
        type: 'style',
        message: 'Consider using array literals [] or object literals {} instead of constructors',
        severity: 'info'
      });
    }
  }
  
  return {
    statistics: {
      lineCount,
      characterCount,
      averageLineLength: averageLineLength.toFixed(2)
    },
    complexity: {
      score: complexityScore,
      indicators: complexityIndicators
    },
    suggestions
  };
}

module.exports = analyzeCode;
EOL
    
    # Create modify-code.js
    cat > "$MCP_DIR/tools/modify-code.js" << 'EOL'
// Code Modification Tool

const axios = require('axios');
const config = require('../config/server');

/**
 * Modifies code based on instructions
 * @param {string} code - The code to modify
 * @param {string} instructions - Instructions for modifying the code
 * @param {string} language - The programming language of the code
 * @returns {Promise<object>} - Modified code and explanation
 */
async function modifyCode(code, instructions, language = 'javascript') {
  // This is a placeholder implementation
  // In a real implementation, you might use an AI service to modify the code
  
  // For now, we'll just implement a few simple transformations
  let modifiedCode = code;
  let explanation = 'Applied the following changes:\n';
  
  // Simple transformations based on instructions
  if (instructions.toLowerCase().includes('add comments')) {
    // Add some placeholder comments
    modifiedCode = addComments(modifiedCode, language);
    explanation += '- Added comments to the code\n';
  }
  
  if (instructions.toLowerCase().includes('format')) {
    // Simple formatting (just adds consistent indentation)
    modifiedCode = formatCode(modifiedCode, language);
    explanation += '- Formatted the code with consistent indentation\n';
  }
  
  if (instructions.toLowerCase().includes('modernize') && (language === 'javascript' || language === 'typescript')) {
    // Replace var with let/const
    const oldCode = modifiedCode;
    modifiedCode = modifiedCode.replace(/var\s+([a-zA-Z0-9_]+)\s*=/g, 'const $1 =');
    if (oldCode !== modifiedCode) {
      explanation += '- Replaced var with const where appropriate\n';
    }
  }
  
  return {
    original: code,
    modified: modifiedCode,
    explanation
  };
}

/**
 * Adds comments to code
 * @param {string} code - The code to add comments to
 * @param {string} language - The programming language of the code
 * @returns {string} - Code with comments added
 */
function addComments(code, language) {
  const lines = code.split('\n');
  const commentedLines = [];
  
  // Determine comment syntax based on language
  let lineComment = '//';
  if (['python', 'ruby'].includes(language)) {
    lineComment = '#';
  } else if (['html', 'xml'].includes(language)) {
    lineComment = '<!--';
  }
  
  // Add a header comment
  commentedLines.push(`${lineComment} Code with added comments`);
  commentedLines.push('');
  
  // Process each line
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Add comments for function/class declarations
    if (line.match(/function\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)/)) {
      const match = line.match(/function\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)/);
      const name = match[1] || match[2];
      commentedLines.push(`${lineComment} ${match[0]} declaration`);
    }
    
    // Add the original line
    commentedLines.push(line);
    
    // Add comments for control structures
    if (line.match(/if|else|for|while|switch/)) {
      const indentation = line.match(/^\s*/)[0];
      const nextLine = i < lines.length - 1 ? lines[i + 1] : '';
      if (nextLine.includes('{')) {
        commentedLines.push(`${indentation}${lineComment} Start of control block`);
      }
    }
  }
  
  return commentedLines.join('\n');
}

/**
 * Formats code with consistent indentation
 * @param {string} code - The code to format
 * @param {string} language - The programming language of the code
 * @returns {string} - Formatted code
 */
function formatCode(code, language) {
  // This is a very simple formatter that just handles indentation
  // A real formatter would be much more sophisticated
  
  const lines = code.split('\n');
  const formattedLines = [];
  let indentLevel = 0;
  
  for (const line of lines) {
    // Trim whitespace
    const trimmedLine = line.trim();
    
    // Skip empty lines
    if (trimmedLine === '') {
      formattedLines.push('');
      continue;
    }
    
    // Decrease indent for closing braces/brackets
    if (trimmedLine.startsWith('}') || trimmedLine.startsWith(']') || trimmedLine.startsWith(')')) {
      indentLevel = Math.max(0, indentLevel - 1);
    }
    
    // Add the line with proper indentation
    const indentation = '  '.repeat(indentLevel);
    formattedLines.push(indentation + trimmedLine);
    
    // Increase indent for opening braces/brackets
    if (trimmedLine.endsWith('{') || trimmedLine.endsWith('[') || trimmedLine.endsWith('(')) {
      indentLevel++;
    }
    
    // Handle single-line blocks
    if (trimmedLine.includes('{') && trimmedLine.includes('}')) {
      // No change to indent level
    }
  }
  
  return formattedLines.join('\n');
}

module.exports = modifyCode;
EOL
    
    # Create search-code.js
    cat > "$MCP_DIR/tools/search-code.js" << 'EOL'
// Code Search Tool

const fs = require('fs').promises;
const path = require('path');
const config = require('../config/server');

/**
 * Searches code for patterns or specific content
 * @param {string} pattern - The pattern to search for
 * @param {string[]} files - The files to search in
 * @returns {Promise<object>} - Search results
 */
async function searchCode(pattern, files = []) {
  // This is a placeholder implementation
  // In a real implementation, you would search actual files in the workspace
  
  // Validate and sanitize the pattern to prevent regex DoS attacks
  if (!pattern || typeof pattern !== 'string') {
    throw new Error('Search pattern must be a non-empty string');
  }
  
  // Limit pattern length to prevent DoS attacks
  if (pattern.length > 1000) {
    throw new Error('Search pattern is too long (max 1000 characters)');
  }
  
  // Escape special regex characters to prevent injection attacks
  // This is safer than directly using user input in a RegExp constructor
  const sanitizedPattern = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  
  // Create a regex from the sanitized pattern with a timeout
  let regex;
  try {
    // Use a safe regex implementation
    regex = new RegExp(sanitizedPattern, 'g');
    
    // Add a timeout for regex execution to prevent catastrophic backtracking
    // This is a simplified version - in production, use a library like safe-regex
    const startTime = Date.now();
    const timeoutMs = 5000; // 5 second timeout
    
    // Test regex with a small sample to check for performance issues
    const testSample = 'test'.repeat(1000);
    const checkTimeout = () => {
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Regex execution timed out - pattern may be too complex');
      }
    };
    
    // Add timeout check to regex exec
    const originalExec = regex.exec;
    regex.exec = function(str) {
      checkTimeout();
      return originalExec.call(this, str);
    };
  } catch (error) {
    throw new Error(`Invalid search pattern: ${error.message}`);
  }
  
  const results = [];
  
  // If no files are specified, use some sample files
  if (!files || files.length === 0) {
    files = [
      'src/index.js',
      'src/components/App.js',
      'src/utils/helpers.js'
    ];
  }
  
  // Sample content for demonstration purposes
  const sampleContent = {
    'src/index.js': `
import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/App';
import './styles/index.css';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
    `,
    'src/components/App.js': `
import React, { useState, useEffect } from 'react';
import { getUser } from '../utils/api';
import { formatDate } from '../utils/helpers';
import Header from './Header';
import Footer from './Footer';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    async function fetchUser() {
      try {
        const userData = await getUser();
        setUser(userData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    
    fetchUser();
  }, []);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  
  return (
    <div className="app">
      <Header user={user} />
      <main>
        <h1>Welcome, {user.name}!</h1>
        <p>Last login: {formatDate(user.lastLogin)}</p>
      </main>
      <Footer />
    </div>
  );
}

export default App;
    `,
    'src/utils/helpers.js': `
/**
 * Formats a date string
 * @param {string} dateString - The date string to format
 * @returns {string} - Formatted date string
 */
export function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

/**
 * Truncates a string to a specified length
 * @param {string} str - The string to truncate
 * @param {number} length - The maximum length
 * @returns {string} - Truncated string
 */
export function truncate(str, length = 100) {
  if (str.length <= length) return str;
  return str.slice(0, length) + '...';
}

/**
 * Debounces a function
 * @param {Function} func - The function to debounce
 * @param {number} wait - The debounce wait time in milliseconds
 * @returns {Function} - Debounced function
 */
export function debounce(func, wait = 300) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}
    `
  };
  
  // Search each file
  for (const file of files) {
    // Get the content (in a real implementation, this would read from the file system)
    const content = sampleContent[file] || '';
    
    // Find all matches
    const matches = [];
    let match;
    while ((match = regex.exec(content)) !== null) {
      // Get the line number and context
      const lines = content.substring(0, match.index).split('\n');
      const lineNumber = lines.length;
      
      // Get context (3 lines before and after)
      const allLines = content.split('\n');
      const startLine = Math.max(0, lineNumber - 3);
      const endLine = Math.min(allLines.length, lineNumber + 3);
      const context = allLines.slice(startLine, endLine).join('\n');
      
      matches.push({
        line: lineNumber,
        column: match.index - lines.slice(0, -1).join('\n').length,
        match: match[0],
        context
      });
    }
    
    if (matches.length > 0) {
      results.push({
        file,
        matches
      });
    }
  }
  
  return {
    pattern,
    files,
    results,
    totalMatches: results.reduce((sum, file) => sum + file.matches.length, 0)
  };
}

module.exports = searchCode;
EOL
    
    print_success "MCP server tool files created."
}

# Install dependencies
install_dependencies() {
    print_info "Installing MCP server dependencies..."
    
    cd "$MCP_DIR"
    npm install
    cd -
    
    print_success "MCP server dependencies installed."
}

# Generate security credentials
generate_security_credentials() {
    print_info "Generating security credentials..."
    
    # Create certs directory
    mkdir -p "$MCP_DIR/certs"
    
    # Generate self-signed certificates
    print_info "Generating self-signed certificates..."
    
    # Create a secure OpenSSL configuration
    cat > "$MCP_DIR/certs/openssl.cnf" << 'EOL'
[req]
default_bits = 4096
default_md = sha256
distinguished_name = req_distinguished_name
prompt = no
encrypt_key = no
x509_extensions = v3_req

[req_distinguished_name]
C = US
ST = California
L = San Francisco
O = SPARC IDE
OU = Development
CN = localhost

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOL
    
    # Generate private key and certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
        -keyout "$MCP_DIR/certs/key.pem" \
        -out "$MCP_DIR/certs/cert.pem" \
        -config "$MCP_DIR/certs/openssl.cnf"
    
    # Set secure permissions
    chmod 600 "$MCP_DIR/certs/key.pem"
    chmod 644 "$MCP_DIR/certs/cert.pem"
    
    # Generate a secure random secret for authentication
    print_info "Generating authentication secret..."
    AUTH_SECRET=$(openssl rand -hex 32)
    
    # Update .env file with the generated secret
    sed -i "s/AUTH_SECRET=/AUTH_SECRET=$AUTH_SECRET/" "$MCP_DIR/.env"
    
    print_success "Security credentials generated successfully."
    print_info "Self-signed certificates created at: $MCP_DIR/certs/"
    print_info "Authentication secret added to .env file."
}

# Create start script
create_start_script() {
    print_info "Creating MCP server start script..."
    cat > "$MCP_DIR/start-mcp-server.sh" << 'EOL'
#!/bin/bash
# Start the MCP server with security checks

set -e
set -o nounset
set -o pipefail

cd "$(dirname "$0")"

# Check if certificates exist
if [ ! -f "./certs/key.pem" ] || [ ! -f "./certs/cert.pem" ]; then
    echo "Error: HTTPS certificates not found. Please run setup-mcp-server.sh again."
    exit 1
fi

# Check if AUTH_SECRET is set in .env
if grep -q "AUTH_SECRET=$" .env || grep -q "AUTH_SECRET=" .env; then
    echo "Warning: Authentication secret not set in .env file."
    echo "Generating a new random secret..."
    AUTH_SECRET=$(openssl rand -hex 32)
    sed -i "s/AUTH_SECRET=.*/AUTH_SECRET=$AUTH_SECRET/" .env
    echo "Authentication secret updated."
fi

# Check if admin password hash is set
if grep -q "MCP_ADMIN_PASSWORD_HASH=$" .env || grep -q "MCP_ADMIN_PASSWORD_HASH=" .env; then
    echo "Warning: Admin password hash not set in .env file."
    echo "Please run ./scripts/generate-admin-password.sh to create a secure password hash"
    echo "and add it to your .env file before starting the server."
    echo "For security reasons, this cannot be auto-generated."
    exit 1
fi

# Check permissions on sensitive files
if [ "$(stat -c %a ./certs/key.pem)" != "600" ]; then
    echo "Fixing permissions on private key..."
    chmod 600 ./certs/key.pem
fi

# Verify .env file is not world-readable
if [ "$(stat -c %a .env)" != "600" ] && [ "$(stat -c %a .env)" != "640" ]; then
    echo "Fixing permissions on .env file..."
    chmod 600 .env
fi

echo "Security checks passed. Starting MCP server..."
npm start
EOL
EOL
    
    chmod +x "$MCP_DIR/start-mcp-server.sh"
    
    print_success "MCP server start script created."
}

# Main function
main() {
    print_info "Setting up SPARC IDE MCP server..."
    
    check_prerequisites
    create_directory_structure
    create_configuration
    create_main_file
    create_tool_files
    install_dependencies
    generate_security_credentials
    create_start_script
    
    print_success "SPARC IDE MCP server set up successfully with enhanced security."
    print_info "HTTPS and authentication are enabled by default."
    print_info "To generate a secure admin password hash, run: ./scripts/generate-admin-password.sh"
    print_info "Add the generated hash to your .env file in the $MCP_DIR directory."
    print_info "To start the MCP server, run: $MCP_DIR/start-mcp-server.sh"
    print_info "Access the server at: https://localhost:$MCP_SERVER_PORT"
}

# Run main function
main