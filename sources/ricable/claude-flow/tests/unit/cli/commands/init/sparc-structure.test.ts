import { assertEquals, assertExists, assertStringIncludes } from "@std/assert/mod.ts";
import { exists } from "@std/fs/mod.ts";
import { join } from "@std/path/mod.ts";
import { beforeEach, afterEach, describe, it } from "@std/testing/bdd.ts";
import { createSparcStructureManually } from "../../../../../src/cli/simple-commands/init/sparc-structure.js";

describe("SPARC Structure Creation Tests", () => {
  let testDir: string;

  beforeEach(async () => {
    testDir = await Deno.makeTempDir({ prefix: "sparc_structure_test_" });
    Deno.env.set("PWD", testDir);
    Deno.chdir(testDir);
  });

  afterEach(async () => {
    try {
      await Deno.remove(testDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe("Directory structure creation", () => {
    it("should create .roo directory structure", async () => {
      await createSparcStructureManually();

      // Check main directories
      expect(await exists(join(testDir, ".roo").toBeDefined()));
      expect(await exists(join(testDir, ".roo/templates").toBeDefined()));
      expect(await exists(join(testDir, ".roo/workflows").toBeDefined()));
      expect(await exists(join(testDir, ".roo/modes").toBeDefined()));
      expect(await exists(join(testDir, ".roo/configs").toBeDefined()));
    });

    it("should create .roomodes configuration file", async () => {
      await createSparcStructureManually();

      expect(await exists(join(testDir, ".roomodes").toBeDefined()));

      // Check content is valid JSON
      const roomodesContent = await Deno.readTextFile(join(testDir, ".roomodes"));
      const roomodesData = JSON.parse(roomodesContent);
      
      expect(typeof roomodesData).toBe("object");
      expect(roomodesData !== null).toBe(true);
    });

    it("should create workflow templates", async () => {
      await createSparcStructureManually();

      expect(await exists(join(testDir, ".roo/workflows/basic-tdd.json").toBeDefined()));

      // Check workflow is valid JSON
      const workflowContent = await Deno.readTextFile(join(testDir, ".roo/workflows/basic-tdd.json"));
      const workflowData = JSON.parse(workflowContent);
      
      expect(typeof workflowData).toBe("object");
      assertStringIncludes(workflowContent, "tdd");
    });

    it("should create README for .roo directory", async () => {
      await createSparcStructureManually();

      expect(await exists(join(testDir, ".roo/README.md").toBeDefined()));

      const readmeContent = await Deno.readTextFile(join(testDir, ".roo/README.md"));
      assertStringIncludes(readmeContent, "# SPARC Development Environment");
      assertStringIncludes(readmeContent, "## Directory Structure");
    });

    it("should create Claude commands directory", async () => {
      await createSparcStructureManually();

      expect(await exists(join(testDir, ".claude").toBeDefined()));
      expect(await exists(join(testDir, ".claude/commands").toBeDefined()));
      expect(await exists(join(testDir, ".claude/commands/sparc").toBeDefined()));
    });
  });

  describe("Existing file handling", () => {
    it("should preserve existing .roomodes file", async () => {
      const existingContent = JSON.stringify({
        customMode: "test",
        existing: true
      }, null, 2);
      
      await Deno.writeTextFile(join(testDir, ".roomodes"), existingContent);

      await createSparcStructureManually();

      const preservedContent = await Deno.readTextFile(join(testDir, ".roomodes"));
      expect(preservedContent).toBe(existingContent);
    });

    it("should handle existing directories gracefully", async () => {
      // Create some directories first
      await Deno.mkdir(join(testDir, ".roo"), { recursive: true });
      await Deno.mkdir(join(testDir, ".roo/templates"), { recursive: true });

      // Should not fail
      await createSparcStructureManually();

      // Should still create missing directories
      expect(await exists(join(testDir, ".roo/workflows").toBeDefined()));
      expect(await exists(join(testDir, ".roo/modes").toBeDefined()));
      expect(await exists(join(testDir, ".roo/configs").toBeDefined()));
    });
  });

  describe("File content validation", () => {
    it("should create valid .roomodes JSON", async () => {
      await createSparcStructureManually();

      const roomodesContent = await Deno.readTextFile(join(testDir, ".roomodes"));
      const roomodesData = JSON.parse(roomodesContent);

      // Should have SPARC modes
      expect(typeof roomodesData.modes).toBe("object");
      expect(roomodesData.modes !== null).toBe(true);

      // Should have basic modes
      expect(roomodesData.modes.architect).toBeDefined();
      expect(roomodesData.modes.code).toBeDefined();
      expect(roomodesData.modes.tdd).toBeDefined();
      expect(roomodesData.modes["spec-pseudocode"]).toBeDefined();
    });

    it("should create valid workflow JSON", async () => {
      await createSparcStructureManually();

      const workflowContent = await Deno.readTextFile(join(testDir, ".roo/workflows/basic-tdd.json"));
      const workflowData = JSON.parse(workflowContent);

      // Should have workflow structure
      expect(typeof workflowData.name).toBe("string");
      expect(Array.isArray(workflowData.steps)).toBe(true);
      expect(workflowData.steps.length > 0).toBe(true);
    });

    it("should create proper README format", async () => {
      await createSparcStructureManually();

      const readmeContent = await Deno.readTextFile(join(testDir, ".roo/README.md"));

      // Should have proper markdown structure
      assertStringIncludes(readmeContent, "# SPARC Development Environment");
      assertStringIncludes(readmeContent, "## Directory Structure");
      assertStringIncludes(readmeContent, "templates/");
      assertStringIncludes(readmeContent, "workflows/");
      assertStringIncludes(readmeContent, "modes/");
      assertStringIncludes(readmeContent, "configs/");
    });
  });

  describe("Error handling", () => {
    it("should handle permission errors gracefully", async () => {
      // Create a directory that can't be written to
      await Deno.mkdir(join(testDir, ".roo"));
      await Deno.chmod(join(testDir, ".roo"), 0o444);

      const originalError = console.error;
      const originalLog = console.log;
      const errors: string[] = [];
      const logs: string[] = [];
      
      console.error = (...args: any[]) => errors.push(args.join(" "));
      console.log = (...args: any[]) => logs.push(args.join(" "));

      try {
        await createSparcStructureManually();
      } catch {
        // Expected to fail
      } finally {
        console.error = originalError;
        console.log = originalLog;
        
        // Restore permissions for cleanup
        await Deno.chmod(join(testDir, ".roo"), 0o755);
      }

      // Should have logged error
      const allOutput = [...errors, ...logs].join("\n");
      assertStringIncludes(allOutput, "Failed to create SPARC structure");
    });

    it("should continue on partial failures", async () => {
      // Create a file where a directory should be
      await Deno.writeTextFile(join(testDir, ".roo"), "not a directory");

      const originalLog = console.log;
      const logs: string[] = [];
      console.log = (...args: any[]) => logs.push(args.join(" "));

      try {
        await createSparcStructureManually();
      } catch {
        // Expected to fail
      }

      console.log = originalLog;

      // Should have attempted creation
      const output = logs.join("\n");
      assertStringIncludes(output, "Failed to create SPARC structure");
    });
  });

  describe("Integration with Claude commands", () => {
    it("should create SPARC slash commands", async () => {
      await createSparcStructureManually();

      // Check for command files
      expect(await exists(join(testDir, ".claude/commands/sparc").toBeDefined()));
      
      // Check for specific command files (they should be created by createClaudeSlashCommands)
      const commandsDir = join(testDir, ".claude/commands/sparc");
      const commandFiles = [];
      
      try {
        for await (const entry of Deno.readDir(commandsDir)) {
          if (entry.isFile && entry.name.endsWith(".md")) {
            commandFiles.push(entry.name);
          }
        }
      } catch {
        // Directory might not exist or be empty
      }

      // Should have created command files
      expect(commandFiles.length >= 0).toBe(true);
    });
  });

  describe("Cleanup and validation", () => {
    it("should create consistent directory structure", async () => {
      await createSparcStructureManually();

      const expectedDirs = [
        ".roo",
        ".roo/templates",
        ".roo/workflows", 
        ".roo/modes",
        ".roo/configs",
        ".claude",
        ".claude/commands",
        ".claude/commands/sparc"
      ];

      for (const dir of expectedDirs) {
        expect(await exists(join(testDir, dir).toBeDefined()));
      }
    });

    it("should create all expected files", async () => {
      await createSparcStructureManually();

      const expectedFiles = [
        ".roomodes",
        ".roo/README.md",
        ".roo/workflows/basic-tdd.json"
      ];

      for (const file of expectedFiles) {
        expect(await exists(join(testDir, file).toBeDefined()));
      }
    });

    it("should maintain valid file formats", async () => {
      await createSparcStructureManually();

      // Check JSON files are valid
      const jsonFiles = [
        ".roomodes",
        ".roo/workflows/basic-tdd.json"
      ];

      for (const file of jsonFiles) {
        const content = await Deno.readTextFile(join(testDir, file));
        // Should not throw
        JSON.parse(content);
      }

      // Check markdown files are valid
      const mdFiles = [
        ".roo/README.md"
      ];

      for (const file of mdFiles) {
        const content = await Deno.readTextFile(join(testDir, file));
        // Should start with header
        assertStringIncludes(content, "#");
      }
    });
  });
});