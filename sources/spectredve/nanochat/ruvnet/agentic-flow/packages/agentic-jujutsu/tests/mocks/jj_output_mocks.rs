//! Mock data for jj command outputs

/// Mock operation log output from `jj op log`
pub const MOCK_OP_LOG: &str = r#"@  qpvuntsm test@example.com 2024-01-01 12:00:00.000 -08:00 abc123
│  describe commit
◉  sqpuoqvx test@example.com 2024-01-01 11:00:00.000 -08:00 def456
│  rebase
◉  yzpnroqq admin@example.org 2024-01-01 10:00:00.000 -08:00 ghi789
│  new empty commit
◉  mzvwutvl admin@example.org 2024-01-01 09:00:00.000 -08:00 jkl012
   initialize repo
"#;

/// Mock status output from `jj status`
pub const MOCK_STATUS_CLEAN: &str = r#"The working copy is clean
Working copy : qpvuntsm 12345678 (no description set)
Parent commit: sqpuoqvx 23456789 Update README
"#;

/// Mock status with changes
pub const MOCK_STATUS_WITH_CHANGES: &str = r#"Working copy changes:
M src/lib.rs
M Cargo.toml
A tests/new_test.rs
D old_file.rs

Working copy : qpvuntsm 12345678 (no description set)
Parent commit: sqpuoqvx 23456789 Update README
"#;

/// Mock conflict output
pub const MOCK_CONFLICTS: &str = r#"Conflict in src/main.rs:
  1: <<<<<<< local
  2:     let x = 1;
  3: =======
  4:     let x = 2;
  5: >>>>>>> remote

Conflict in Cargo.toml:
  1: <<<<<<< local
  2: version = "0.1.0"
  3: =======
  4: version = "0.2.0"
  5: >>>>>>> remote
"#;

/// Mock log output from `jj log`
pub const MOCK_LOG: &str = r#"@  qpvuntsm test@example.com 2024-01-05 12:00:00
│  feat: Add new feature
│
◉  sqpuoqvx test@example.com 2024-01-04 11:00:00
│  fix: Bug fix
│
◉  yzpnroqq admin@example.org 2024-01-03 10:00:00
│  docs: Update documentation
│
◉  mzvwutvl admin@example.org 2024-01-02 09:00:00
   chore: Initial commit
"#;

/// Mock branch list output
pub const MOCK_BRANCHES: &str = r#"main: sqpuoqvx 12345678 Update README
feature-1: qpvuntsm 23456789 Add feature
feature-2: yzpnroqq 34567890 Fix bug
"#;

/// Mock show output for a commit
pub const MOCK_SHOW_COMMIT: &str = r#"Commit ID: abc123def456
Change ID: qpvuntsm12345678
Author: Test User <test@example.com>
Date: 2024-01-01 12:00:00 -08:00

    feat: Add new feature

    This commit adds a new feature to the project.

diff --git a/src/lib.rs b/src/lib.rs
index abc123..def456 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,6 @@
+pub fn new_feature() {
+    println!("New feature!");
+}
+
 pub fn hello() {
     println!("Hello, world!");
 }
"#;

/// Mock error output when jj command fails
pub const MOCK_ERROR_NOT_A_REPO: &str = r#"Error: Not a jujutsu repo
There is no jujutsu repo in "." or any of its parents
"#;

/// Mock error for conflict
pub const MOCK_ERROR_CONFLICT: &str = r#"Error: The working copy has conflicts
Resolve the conflicts and run `jj resolve` to continue
"#;

/// Mock output from `jj init`
pub const MOCK_INIT_OUTPUT: &str = r#"Initialized repo in .
"#;

/// Mock output from successful commit
pub const MOCK_COMMIT_SUCCESS: &str = r#"Working copy now at: qpvuntsm 12345678 feat: Add feature
Parent commit      : sqpuoqvx 23456789 Previous commit
"#;

/// Mock output from rebase
pub const MOCK_REBASE_SUCCESS: &str = r#"Rebased 3 commits
Working copy now at: qpvuntsm 12345678 (no description set)
"#;

/// Mock output from merge with conflicts
pub const MOCK_MERGE_WITH_CONFLICTS: &str = r#"New conflicts appeared in these commits:
  qpvuntsm 12345678 (conflict) Merge commit
To resolve the conflicts, start by updating to it:
  jj new qpvuntsm
"#;

/// Mock diff output
pub const MOCK_DIFF: &str = r#"diff --git a/src/main.rs b/src/main.rs
index abc123..def456 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,5 +1,8 @@
 fn main() {
-    println!("Hello, world!");
+    println!("Hello, Jujutsu!");
+
+    // New feature
+    new_feature();
 }

+fn new_feature() {}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_data_not_empty() {
        assert!(!MOCK_OP_LOG.is_empty());
        assert!(!MOCK_STATUS_CLEAN.is_empty());
        assert!(!MOCK_CONFLICTS.is_empty());
    }

    #[test]
    fn test_mock_op_log_has_operations() {
        assert!(MOCK_OP_LOG.contains("qpvuntsm"));
        assert!(MOCK_OP_LOG.contains("test@example.com"));
    }

    #[test]
    fn test_mock_status_variants() {
        assert!(MOCK_STATUS_CLEAN.contains("clean"));
        assert!(MOCK_STATUS_WITH_CHANGES.contains("M src/lib.rs"));
    }

    #[test]
    fn test_mock_conflicts_format() {
        assert!(MOCK_CONFLICTS.contains("<<<<<<<"));
        assert!(MOCK_CONFLICTS.contains(">>>>>>>"));
    }
}
