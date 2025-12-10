#!/bin/bash
# SPARC IDE - UI Configuration Script
# This script configures the UI layout and appearance for SPARC IDE

set -e
set -o nounset  # Exit if a variable is unset
set -o pipefail # Exit if any command in a pipeline fails

# Configuration
VSCODIUM_DIR="vscodium"
CONFIG_DIR="src/config"
WORKBENCH_CONFIG_PATH="src/vs/workbench/browser/workbench.js"
LAYOUT_CONFIG_PATH="src/vs/workbench/browser/parts/layout.js"
THEME_CONFIG_PATH="src/vs/platform/theme/common/colorRegistry.js"

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

# Check if VSCodium directory exists and validate its integrity
check_vscodium() {
    print_info "Checking VSCodium directory..."
    
    if [ ! -d "$VSCODIUM_DIR" ]; then
        print_error "VSCodium directory not found. Please run setup-build-environment.sh first."
        exit 1
    fi
    
    # Verify that it's a valid VSCodium directory
    if [ ! -f "$VSCODIUM_DIR/package.json" ]; then
        print_error "Invalid VSCodium directory. package.json not found."
        exit 1
    fi
    
    # Check for suspicious files
    print_info "Checking for suspicious files in VSCodium directory..."
    if find "$VSCODIUM_DIR" -name "*.sh" -o -name "*.bash" | xargs grep -l "curl.*sh.*|.*bash" > /dev/null 2>&1; then
        print_error "Suspicious scripts found that may download and execute code. Security check failed."
        exit 1
    fi
    
    # Verify permissions
    if [ -d "$VSCODIUM_DIR/.git" ] && [ "$(stat -c %a "$VSCODIUM_DIR/.git")" != "755" ]; then
        print_info "Fixing permissions on .git directory..."
        chmod 755 "$VSCODIUM_DIR/.git"
    fi
    
    print_success "VSCodium directory found and validated."
}

# Configure AI-centric layout
configure_layout() {
    print_info "Configuring AI-centric layout..."
    
    # Create a temporary directory for verification
    TEMP_DIR=$(mktemp -d)
    print_info "Created temporary directory for verification: $TEMP_DIR"
    
    # Create layout patch file in temporary directory
    LAYOUT_PATCH_FILE="$TEMP_DIR/ai-centric-layout.patch"
    
    cat > "$LAYOUT_PATCH_FILE" << 'EOL'
--- a/src/vs/workbench/browser/parts/layout.js
+++ b/src/vs/workbench/browser/parts/layout.js
@@ -123,6 +123,7 @@
     // Layout configurations
     this.layoutConfigurations = {
         'default': { sidebarPosition: 'left', panelPosition: 'bottom', activityBarVisible: true },
+        'ai-centric': { sidebarPosition: 'left', panelPosition: 'bottom', activityBarVisible: true, sidebarWidth: 300, panelHeight: 250 },
         'zen': { sidebarPosition: 'hidden', panelPosition: 'hidden', activityBarVisible: false },
         'minimal': { sidebarPosition: 'left', panelPosition: 'bottom', activityBarVisible: false }
     };
@@ -130,7 +131,7 @@
     // Default layout configuration
-    this.defaultLayoutConfig = 'default';
+    this.defaultLayoutConfig = 'ai-centric';
     
     // Initialize layout
     this.initLayout();
EOL
    
    # Verify patch file for security
    if grep -q "curl\|wget\|eval\|exec" "$LAYOUT_PATCH_FILE"; then
        print_error "Suspicious content found in patch file. Security check failed."
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    # Set secure permissions
    chmod 644 "$LAYOUT_PATCH_FILE"
    
    # Create secure patches directory
    mkdir -p "build/patches"
    chmod 755 "build/patches"
    
    # Move verified patch to final location
    cp "$LAYOUT_PATCH_FILE" "build/patches/ai-centric-layout.patch"
    
    # Apply layout patch if VSCodium directory exists
    if [ -d "$VSCODIUM_DIR" ] && [ -f "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH" ]; then
        # Verify target file exists and is writable
        if [ ! -w "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH" ]; then
            print_error "Target file is not writable: $VSCODIUM_DIR/$LAYOUT_CONFIG_PATH"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        # Create backup of original file
        cp "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH" "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH.bak"
        
        # Apply patch with error handling
        if ! patch -N "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH" < "build/patches/ai-centric-layout.patch"; then
            print_error "Failed to apply layout patch. Restoring backup."
            cp "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH.bak" "$VSCODIUM_DIR/$LAYOUT_CONFIG_PATH"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        print_success "AI-centric layout configured."
    else
        print_info "Layout configuration will be applied during build."
    fi
    
    # Clean up
    rm -rf "$TEMP_DIR"
}

# Configure custom themes
configure_themes() {
    print_info "Configuring custom themes..."
    
    # Create themes directory
    mkdir -p "src/themes"
    
    # Create Dracula Pro theme
    cat > "src/themes/dracula-pro.json" << 'EOL'
{
  "name": "Dracula Pro",
  "type": "dark",
  "colors": {
    "editor.background": "#22212C",
    "editor.foreground": "#F8F8F2",
    "activityBarBadge.background": "#FF80BF",
    "sideBarTitle.foreground": "#F8F8F2",
    "statusBar.background": "#454158",
    "statusBar.noFolderBackground": "#454158",
    "statusBar.debuggingBackground": "#FF80BF",
    "list.activeSelectionBackground": "#454158",
    "list.inactiveSelectionBackground": "#2D2B38",
    "list.hoverBackground": "#2D2B38",
    "list.dropBackground": "#454158",
    "list.highlightForeground": "#8AFF80",
    "editorLineNumber.foreground": "#6272A4",
    "editorLineNumber.activeForeground": "#F8F8F2",
    "editorCursor.foreground": "#F8F8F2",
    "editor.selectionBackground": "#454158",
    "editor.selectionHighlightBackground": "#424450",
    "editor.wordHighlightBackground": "#8F43EB80",
    "editor.wordHighlightStrongBackground": "#8F43EB80",
    "editor.findMatchBackground": "#FFB86C80",
    "editor.findMatchHighlightBackground": "#FFB86C40",
    "editor.findRangeHighlightBackground": "#8F43EB30",
    "editor.hoverHighlightBackground": "#8F43EB50",
    "editor.lineHighlightBackground": "#2D2B38",
    "editor.lineHighlightBorder": "#2D2B38",
    "editorLink.activeForeground": "#8BE9FD",
    "editor.rangeHighlightBackground": "#8F43EB30",
    "editorWhitespace.foreground": "#424450",
    "editorIndentGuide.background": "#424450",
    "editorIndentGuide.activeBackground": "#6272A4",
    "editorRuler.foreground": "#424450",
    "editorCodeLens.foreground": "#6272A4",
    "editorBracketMatch.background": "#8F43EB30",
    "editorBracketMatch.border": "#8F43EB",
    "editorOverviewRuler.border": "#2D2B38",
    "editorOverviewRuler.findMatchForeground": "#FFB86C",
    "editorOverviewRuler.rangeHighlightForeground": "#8F43EB",
    "editorOverviewRuler.selectionHighlightForeground": "#8BE9FD",
    "editorOverviewRuler.wordHighlightForeground": "#8F43EB",
    "editorOverviewRuler.wordHighlightStrongForeground": "#8F43EB",
    "editorOverviewRuler.modifiedForeground": "#8BE9FD80",
    "editorOverviewRuler.addedForeground": "#8AFF8080",
    "editorOverviewRuler.deletedForeground": "#FF80BF80",
    "editorOverviewRuler.errorForeground": "#FF5555",
    "editorOverviewRuler.warningForeground": "#FFB86C",
    "editorOverviewRuler.infoForeground": "#8BE9FD",
    "editorGutter.modifiedBackground": "#8BE9FD80",
    "editorGutter.addedBackground": "#8AFF8080",
    "editorGutter.deletedBackground": "#FF80BF80",
    "diffEditor.insertedTextBackground": "#8AFF8020",
    "diffEditor.removedTextBackground": "#FF80BF20",
    "scrollbar.shadow": "#22212C",
    "scrollbarSlider.background": "#45415880",
    "scrollbarSlider.hoverBackground": "#454158A0",
    "scrollbarSlider.activeBackground": "#454158D0",
    "panel.background": "#22212C",
    "panel.border": "#2D2B38",
    "panelTitle.activeBorder": "#FF80BF",
    "panelTitle.activeForeground": "#F8F8F2",
    "panelTitle.inactiveForeground": "#6272A4",
    "badge.background": "#FF80BF",
    "badge.foreground": "#F8F8F2",
    "terminal.background": "#22212C",
    "terminal.foreground": "#F8F8F2",
    "terminal.ansiBlack": "#22212C",
    "terminal.ansiRed": "#FF5555",
    "terminal.ansiGreen": "#8AFF80",
    "terminal.ansiYellow": "#FFB86C",
    "terminal.ansiBlue": "#9580FF",
    "terminal.ansiMagenta": "#FF80BF",
    "terminal.ansiCyan": "#8BE9FD",
    "terminal.ansiWhite": "#F8F8F2",
    "terminal.ansiBrightBlack": "#6272A4",
    "terminal.ansiBrightRed": "#FF6E6E",
    "terminal.ansiBrightGreen": "#A2FF99",
    "terminal.ansiBrightYellow": "#FFCA85",
    "terminal.ansiBrightBlue": "#B0A2FF",
    "terminal.ansiBrightMagenta": "#FFA5D2",
    "terminal.ansiBrightCyan": "#A4FFFF",
    "terminal.ansiBrightWhite": "#FFFFFF"
  },
  "tokenColors": [
    {
      "scope": ["comment", "punctuation.definition.comment"],
      "settings": {
        "foreground": "#6272A4"
      }
    },
    {
      "scope": ["string", "punctuation.definition.string"],
      "settings": {
        "foreground": "#8AFF80"
      }
    },
    {
      "scope": ["constant.numeric"],
      "settings": {
        "foreground": "#9580FF"
      }
    },
    {
      "scope": ["constant.language"],
      "settings": {
        "foreground": "#FF80BF"
      }
    },
    {
      "scope": ["keyword"],
      "settings": {
        "foreground": "#FF80BF"
      }
    },
    {
      "scope": ["storage"],
      "settings": {
        "foreground": "#FF80BF"
      }
    },
    {
      "scope": ["entity.name.function"],
      "settings": {
        "foreground": "#8BE9FD"
      }
    },
    {
      "scope": ["variable.parameter"],
      "settings": {
        "foreground": "#FFB86C"
      }
    }
  ]
}
EOL
    
    # Create Material Theme
    cat > "src/themes/material-theme.json" << 'EOL'
{
  "name": "Material Theme",
  "type": "dark",
  "colors": {
    "editor.background": "#263238",
    "editor.foreground": "#EEFFFF",
    "activityBarBadge.background": "#80CBC4",
    "sideBarTitle.foreground": "#EEFFFF",
    "statusBar.background": "#1E272C",
    "statusBar.noFolderBackground": "#1E272C",
    "statusBar.debuggingBackground": "#C792EA",
    "list.activeSelectionBackground": "#1E272C",
    "list.inactiveSelectionBackground": "#2E3C43",
    "list.hoverBackground": "#2E3C43",
    "list.dropBackground": "#1E272C",
    "list.highlightForeground": "#80CBC4",
    "editorLineNumber.foreground": "#546E7A",
    "editorLineNumber.activeForeground": "#EEFFFF",
    "editorCursor.foreground": "#FFCC00",
    "editor.selectionBackground": "#1E272C",
    "editor.selectionHighlightBackground": "#314549",
    "editor.wordHighlightBackground": "#314549",
    "editor.wordHighlightStrongBackground": "#314549",
    "editor.findMatchBackground": "#F78C6C80",
    "editor.findMatchHighlightBackground": "#F78C6C40",
    "editor.findRangeHighlightBackground": "#C792EA30",
    "editor.hoverHighlightBackground": "#C792EA50",
    "editor.lineHighlightBackground": "#2E3C43",
    "editor.lineHighlightBorder": "#2E3C43",
    "editorLink.activeForeground": "#80CBC4",
    "editor.rangeHighlightBackground": "#C792EA30",
    "editorWhitespace.foreground": "#546E7A",
    "editorIndentGuide.background": "#37474F",
    "editorIndentGuide.activeBackground": "#546E7A",
    "editorRuler.foreground": "#37474F",
    "editorCodeLens.foreground": "#546E7A",
    "editorBracketMatch.background": "#C792EA30",
    "editorBracketMatch.border": "#C792EA",
    "editorOverviewRuler.border": "#2E3C43",
    "editorOverviewRuler.findMatchForeground": "#F78C6C",
    "editorOverviewRuler.rangeHighlightForeground": "#C792EA",
    "editorOverviewRuler.selectionHighlightForeground": "#80CBC4",
    "editorOverviewRuler.wordHighlightForeground": "#C792EA",
    "editorOverviewRuler.wordHighlightStrongForeground": "#C792EA",
    "editorOverviewRuler.modifiedForeground": "#80CBC480",
    "editorOverviewRuler.addedForeground": "#C3E88D80",
    "editorOverviewRuler.deletedForeground": "#F0719880",
    "editorOverviewRuler.errorForeground": "#FF5370",
    "editorOverviewRuler.warningForeground": "#FFCB6B",
    "editorOverviewRuler.infoForeground": "#82AAFF",
    "editorGutter.modifiedBackground": "#80CBC480",
    "editorGutter.addedBackground": "#C3E88D80",
    "editorGutter.deletedBackground": "#F0719880",
    "diffEditor.insertedTextBackground": "#C3E88D20",
    "diffEditor.removedTextBackground": "#F0719820",
    "scrollbar.shadow": "#263238",
    "scrollbarSlider.background": "#37474F80",
    "scrollbarSlider.hoverBackground": "#37474FA0",
    "scrollbarSlider.activeBackground": "#37474FD0",
    "panel.background": "#263238",
    "panel.border": "#2E3C43",
    "panelTitle.activeBorder": "#80CBC4",
    "panelTitle.activeForeground": "#EEFFFF",
    "panelTitle.inactiveForeground": "#546E7A",
    "badge.background": "#80CBC4",
    "badge.foreground": "#263238",
    "terminal.background": "#263238",
    "terminal.foreground": "#EEFFFF",
    "terminal.ansiBlack": "#263238",
    "terminal.ansiRed": "#FF5370",
    "terminal.ansiGreen": "#C3E88D",
    "terminal.ansiYellow": "#FFCB6B",
    "terminal.ansiBlue": "#82AAFF",
    "terminal.ansiMagenta": "#C792EA",
    "terminal.ansiCyan": "#89DDFF",
    "terminal.ansiWhite": "#EEFFFF",
    "terminal.ansiBrightBlack": "#546E7A",
    "terminal.ansiBrightRed": "#FF5370",
    "terminal.ansiBrightGreen": "#C3E88D",
    "terminal.ansiBrightYellow": "#FFCB6B",
    "terminal.ansiBrightBlue": "#82AAFF",
    "terminal.ansiBrightMagenta": "#C792EA",
    "terminal.ansiBrightCyan": "#89DDFF",
    "terminal.ansiBrightWhite": "#FFFFFF"
  },
  "tokenColors": [
    {
      "scope": ["comment", "punctuation.definition.comment"],
      "settings": {
        "foreground": "#546E7A"
      }
    },
    {
      "scope": ["string", "punctuation.definition.string"],
      "settings": {
        "foreground": "#C3E88D"
      }
    },
    {
      "scope": ["constant.numeric"],
      "settings": {
        "foreground": "#F78C6C"
      }
    },
    {
      "scope": ["constant.language"],
      "settings": {
        "foreground": "#FF5370"
      }
    },
    {
      "scope": ["keyword"],
      "settings": {
        "foreground": "#C792EA"
      }
    },
    {
      "scope": ["storage"],
      "settings": {
        "foreground": "#C792EA"
      }
    },
    {
      "scope": ["entity.name.function"],
      "settings": {
        "foreground": "#82AAFF"
      }
    },
    {
      "scope": ["variable.parameter"],
      "settings": {
        "foreground": "#FFCB6B"
      }
    }
  ]
}
EOL
    
    print_success "Custom themes configured."
}

# Configure SPARC workflow UI
configure_sparc_workflow() {
    print_info "Configuring SPARC workflow UI..."
    
    # Create SPARC workflow UI configuration
    mkdir -p "src/sparc-workflow"
    
    # Create SPARC workflow UI configuration file
    cat > "src/sparc-workflow/config.json" << 'EOL'
{
  "phases": [
    {
      "id": "specification",
      "name": "Specification",
      "description": "Define detailed requirements and acceptance criteria",
      "icon": "document",
      "color": "#42A5F5",
      "templates": ["requirements.md", "user-stories.md", "acceptance-criteria.md"],
      "aiPrompts": ["Generate requirements", "Create user stories", "Define acceptance criteria"]
    },
    {
      "id": "pseudocode",
      "name": "Pseudocode",
      "description": "Create implementation pseudocode and logic flow",
      "icon": "code",
      "color": "#66BB6A",
      "templates": ["pseudocode.md", "flow-diagram.md", "data-structures.md"],
      "aiPrompts": ["Generate pseudocode", "Create flow diagram", "Define data structures"]
    },
    {
      "id": "architecture",
      "name": "Architecture",
      "description": "Design system architecture and component interactions",
      "icon": "package",
      "color": "#FFA726",
      "templates": ["architecture.md", "components.md", "interfaces.md"],
      "aiPrompts": ["Design architecture", "Define components", "Specify interfaces"]
    },
    {
      "id": "refinement",
      "name": "Refinement",
      "description": "Implement iterative improvements and testing",
      "icon": "tools",
      "color": "#EC407A",
      "templates": ["implementation.md", "tests.md", "refactoring.md"],
      "aiPrompts": ["Implement feature", "Write tests", "Refactor code"]
    },
    {
      "id": "completion",
      "name": "Completion",
      "description": "Finalize documentation, deployment, and maintenance",
      "icon": "check",
      "color": "#AB47BC",
      "templates": ["documentation.md", "deployment.md", "maintenance.md"],
      "aiPrompts": ["Generate documentation", "Create deployment plan", "Define maintenance procedures"]
    }
  ],
  "templates": {
    "requirements.md": "# Requirements\n\n## Functional Requirements\n\n- [ ] Requirement 1\n- [ ] Requirement 2\n- [ ] Requirement 3\n\n## Non-Functional Requirements\n\n- [ ] Performance\n- [ ] Security\n- [ ] Usability\n\n## Constraints\n\n- [ ] Constraint 1\n- [ ] Constraint 2\n- [ ] Constraint 3",
    "user-stories.md": "# User Stories\n\n## User Story 1\n\nAs a [user type], I want to [action] so that [benefit].\n\n### Acceptance Criteria\n\n- [ ] Criteria 1\n- [ ] Criteria 2\n- [ ] Criteria 3\n\n## User Story 2\n\nAs a [user type], I want to [action] so that [benefit].\n\n### Acceptance Criteria\n\n- [ ] Criteria 1\n- [ ] Criteria 2\n- [ ] Criteria 3",
    "acceptance-criteria.md": "# Acceptance Criteria\n\n## Feature 1\n\n- [ ] Criteria 1\n- [ ] Criteria 2\n- [ ] Criteria 3\n\n## Feature 2\n\n- [ ] Criteria 1\n- [ ] Criteria 2\n- [ ] Criteria 3",
    "pseudocode.md": "# Pseudocode\n\n```\nFUNCTION main():\n    // Initialize\n    \n    // Process\n    \n    // Output\n    \nEND FUNCTION\n```",
    "flow-diagram.md": "# Flow Diagram\n\n```mermaid\nflowchart TD\n    A[Start] --> B{Decision}\n    B -->|Yes| C[Process 1]\n    B -->|No| D[Process 2]\n    C --> E[End]\n    D --> E\n```",
    "data-structures.md": "# Data Structures\n\n## Structure 1\n\n```\nStructure1 {\n    field1: Type,\n    field2: Type,\n    field3: Type\n}\n```\n\n## Structure 2\n\n```\nStructure2 {\n    field1: Type,\n    field2: Type,\n    field3: Type\n}\n```",
    "architecture.md": "# Architecture\n\n## Components\n\n- Component 1\n- Component 2\n- Component 3\n\n## Interactions\n\n```mermaid\nflowchart LR\n    A[Component 1] --> B[Component 2]\n    B --> C[Component 3]\n    C --> A\n```",
    "components.md": "# Components\n\n## Component 1\n\n- Responsibility: \n- Dependencies: \n- Interfaces: \n\n## Component 2\n\n- Responsibility: \n- Dependencies: \n- Interfaces: ",
    "interfaces.md": "# Interfaces\n\n## Interface 1\n\n```typescript\ninterface Interface1 {\n    method1(): ReturnType;\n    method2(param: Type): ReturnType;\n    property1: Type;\n}\n```\n\n## Interface 2\n\n```typescript\ninterface Interface2 {\n    method1(): ReturnType;\n    method2(param: Type): ReturnType;\n    property1: Type;\n}\n```",
    "implementation.md": "# Implementation\n\n## Component 1\n\n```typescript\nclass Component1 implements Interface1 {\n    // Implementation\n}\n```\n\n## Component 2\n\n```typescript\nclass Component2 implements Interface2 {\n    // Implementation\n}\n```",
    "tests.md": "# Tests\n\n## Unit Tests\n\n```typescript\ndescribe('Component1', () => {\n    test('should do something', () => {\n        // Test implementation\n    });\n});\n```\n\n## Integration Tests\n\n```typescript\ndescribe('Integration', () => {\n    test('should integrate components', () => {\n        // Test implementation\n    });\n});\n```",
    "refactoring.md": "# Refactoring\n\n## Code Smells\n\n- Code smell 1\n- Code smell 2\n- Code smell 3\n\n## Refactoring Steps\n\n1. Step 1\n2. Step 2\n3. Step 3",
    "documentation.md": "# Documentation\n\n## Overview\n\n## Installation\n\n## Usage\n\n## API Reference\n\n## Examples\n\n## Troubleshooting",
    "deployment.md": "# Deployment\n\n## Prerequisites\n\n## Installation\n\n## Configuration\n\n## Verification\n\n## Rollback Plan",
    "maintenance.md": "# Maintenance\n\n## Routine Maintenance\n\n## Monitoring\n\n## Backup and Recovery\n\n## Upgrades\n\n## Support"
  }
}
EOL
    
    print_success "SPARC workflow UI configured."
}

# Main function
main() {
    print_info "Configuring SPARC IDE UI..."
    
    check_vscodium
    configure_layout
    configure_themes
    configure_sparc_workflow
    
    print_success "SPARC IDE UI configured successfully."
}

# Run main function
main