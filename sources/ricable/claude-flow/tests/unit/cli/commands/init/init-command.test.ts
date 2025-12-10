import { assertEquals, assertExists, assertStringIncludes } from "@std/assert/mod.ts";
import { ensureDir, exists } from "@std/fs/mod.ts";
import { join } from "@std/path/mod.ts";
import { beforeEach, afterEach, describe, it } from "@std/testing/bdd.ts";
import { initCommand } from "../../../../../src/cli/simple-commands/init/index.js";

describe("Init Command Unit Tests", () => {
  let testDir: string;

  beforeEach(async () => {
    // Create a temporary test directory
    testDir = await Deno.makeTempDir({ prefix: "claude_flow_init_test_" });
    // Store original working directory
    Deno.env.set("ORIGINAL_CWD", Deno.cwd());
    // Set PWD for init command
    Deno.env.set("PWD", testDir);
    // Change to test directory
    Deno.chdir(testDir);
  });

  afterEach(async () => {
    // Restore original working directory
    const originalCwd = Deno.env.get("ORIGINAL_CWD");
    if (originalCwd) {
      Deno.chdir(originalCwd);
    }
    // Clean up test directory
    try {
      await Deno.remove(testDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe("Basic initialization", () => {
    it("should create basic structure with no flags", async () => {
      await initCommand([], {});

      // Check basic files
      expect(await exists(join(testDir, "CLAUDE.md").toBeDefined()));
      expect(await exists(join(testDir, "memory-bank.md").toBeDefined()));
      expect(await exists(join(testDir, "coordination.md").toBeDefined()));

      // Check directories
      expect(await exists(join(testDir, "memory").toBeDefined()));
      expect(await exists(join(testDir, "memory/agents").toBeDefined()));
      expect(await exists(join(testDir, "memory/sessions").toBeDefined()));
      expect(await exists(join(testDir, "coordination").toBeDefined()));
      expect(await exists(join(testDir, ".claude").toBeDefined()));
      expect(await exists(join(testDir, ".claude/commands").toBeDefined()));
    });

    it("should create minimal structure with --minimal flag", async () => {
      await initCommand(["--minimal"], {});

      // Check files exist
      expect(await exists(join(testDir, "CLAUDE.md").toBeDefined()));
      expect(await exists(join(testDir, "memory-bank.md").toBeDefined()));
      expect(await exists(join(testDir, "coordination.md").toBeDefined()));

      // Check content is minimal (smaller size)
      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      const memoryBankMd = await Deno.readTextFile(join(testDir, "memory-bank.md"));
      
      // Minimal versions should be shorter
      expect(claudeMd.includes("Minimal project configuration")).toBe(true);
      expect(memoryBankMd.includes("Simple memory tracking")).toBe(true);
    });

    it("should handle help flag correctly", async () => {
      const originalLog = console.log;
      const logs: string[] = [];
      console.log = (...args: any[]) => logs.push(args.join(" "));

      await initCommand(["--help"], {});

      console.log = originalLog;

      // Check that help was displayed
      const helpOutput = logs.join("\n");
      assertStringIncludes(helpOutput, "Initialize Claude Code integration");
      assertStringIncludes(helpOutput, "--force");
      assertStringIncludes(helpOutput, "--minimal");
      assertStringIncludes(helpOutput, "--sparc");
    });
  });

  describe("Force flag behavior", () => {
    it("should fail when files exist without --force", async () => {
      // Create existing file
      await Deno.writeTextFile(join(testDir, "CLAUDE.md"), "existing content");

      const originalLog = console.log;
      const logs: string[] = [];
      console.log = (...args: any[]) => logs.push(args.join(" "));

      await initCommand([], {});

      console.log = originalLog;

      // Check warning was displayed
      const output = logs.join("\n");
      assertStringIncludes(output, "already exist");
      assertStringIncludes(output, "Use --force to overwrite");
    });

    it("should overwrite files with --force flag", async () => {
      // Create existing files
      await Deno.writeTextFile(join(testDir, "CLAUDE.md"), "old content");
      await Deno.writeTextFile(join(testDir, "memory-bank.md"), "old memory");

      await initCommand(["--force"], {});

      // Check files were overwritten
      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      const memoryBankMd = await Deno.readTextFile(join(testDir, "memory-bank.md"));

      // Should not contain old content
      expect(claudeMd.includes("old content")).toBe(false);
      expect(memoryBankMd.includes("old memory")).toBe(false);

      // Should contain new content
      assertStringIncludes(claudeMd, "Claude Code Configuration");
      assertStringIncludes(memoryBankMd, "Memory Bank");
    });
  });

  describe("SPARC flag behavior", () => {
    it("should create SPARC-enhanced structure with --sparc flag", async () => {
      await initCommand(["--sparc"], {});

      // Check SPARC-specific files
      expect(await exists(join(testDir, "CLAUDE.md").toBeDefined()));
      expect(await exists(join(testDir, ".claude/commands/sparc").toBeDefined()));

      // Check CLAUDE.md contains SPARC content
      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      assertStringIncludes(claudeMd, "SPARC");
      assertStringIncludes(claudeMd, "Test-Driven Development");
    });

    it("should create SPARC structure manually when create-sparc fails", async () => {
      // This will trigger manual creation since create-sparc won't exist
      await initCommand(["--sparc", "--force"], {});

      // Check manual SPARC structure
      expect(await exists(join(testDir, ".roo").toBeDefined()));
      expect(await exists(join(testDir, ".roo/templates").toBeDefined()));
      expect(await exists(join(testDir, ".roo/workflows").toBeDefined()));
      expect(await exists(join(testDir, ".roomodes").toBeDefined()));
    });
  });

  describe("File content validation", () => {
    it("should create valid JSON files", async () => {
      await initCommand([], {});

      // Check claude-flow-data.json is valid JSON
      const dataPath = join(testDir, "memory/claude-flow-data.json");
      expect(await exists(dataPath).toBeDefined());

      const data = JSON.parse(await Deno.readTextFile(dataPath));
      expect(Array.isArray(data.agents)).toBe(true);
      expect(Array.isArray(data.tasks)).toBe(true);
      expect(typeof data.lastUpdated).toBe("number");
    });

    it("should create proper README files", async () => {
      await initCommand([], {});

      // Check README files
      const agentsReadme = await Deno.readTextFile(join(testDir, "memory/agents/README.md"));
      const sessionsReadme = await Deno.readTextFile(join(testDir, "memory/sessions/README.md"));

      assertStringIncludes(agentsReadme, "# Agent Memory Storage");
      assertStringIncludes(sessionsReadme, "# Session Memory Storage");
    });
  });

  describe("Error handling", () => {
    it("should handle directory creation errors gracefully", async () => {
      // Create a file where a directory should be
      await Deno.writeTextFile(join(testDir, "memory"), "not a directory");

      const originalError = console.error;
      const errors: string[] = [];
      console.error = (...args: any[]) => errors.push(args.join(" "));

      try {
        await initCommand([], {});
      } catch {
        // Expected to fail
      }

      console.error = originalError;

      // Should have attempted to create directory
      expect(errors.length > 0).toBe(true);
    });

    it("should continue when some operations fail", async () => {
      // Make directory read-only (will fail on some operations)
      await Deno.chmod(testDir, 0o555);

      try {
        await initCommand(["--force"], {});
      } catch {
        // Expected some operations to fail
      }

      // Restore permissions
      await Deno.chmod(testDir, 0o755);

      // Should have created at least some files before failing
      // (This test may vary based on OS permissions)
    });
  });

  describe("Flag combinations", () => {
    it("should handle --sparc --minimal combination", async () => {
      await initCommand(["--sparc", "--minimal"], {});

      // Should create SPARC structure
      expect(await exists(join(testDir, ".claude/commands/sparc").toBeDefined()));

      // But with minimal content
      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      assertStringIncludes(claudeMd, "SPARC");
      
      // Memory bank should be minimal
      const memoryBankMd = await Deno.readTextFile(join(testDir, "memory-bank.md"));
      assertStringIncludes(memoryBankMd, "Simple memory tracking");
    });

    it("should handle --sparc --force combination", async () => {
      // Create existing files
      await Deno.writeTextFile(join(testDir, "CLAUDE.md"), "old content");
      await Deno.writeTextFile(join(testDir, ".roomodes"), "old roomodes");

      await initCommand(["--sparc", "--force"], {});

      // Should overwrite and create SPARC structure
      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      assertStringIncludes(claudeMd, "SPARC");
      expect(claudeMd.includes("old content")).toBe(false);

      // Should preserve or recreate .roomodes
      expect(await exists(join(testDir, ".roomodes").toBeDefined()));
    });

    it("should handle all flags together", async () => {
      await initCommand(["--sparc", "--minimal", "--force"], {});

      // Should create minimal SPARC structure
      expect(await exists(join(testDir, "CLAUDE.md").toBeDefined()));
      expect(await exists(join(testDir, ".claude/commands/sparc").toBeDefined()));

      const claudeMd = await Deno.readTextFile(join(testDir, "CLAUDE.md"));
      assertStringIncludes(claudeMd, "SPARC");
    });
  });
});