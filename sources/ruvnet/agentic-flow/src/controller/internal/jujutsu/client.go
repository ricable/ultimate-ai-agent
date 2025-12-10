package jujutsu

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// Client provides access to Jujutsu operations
type Client struct {
	repoPath string
}

// NewClient creates a new Jujutsu client
func NewClient(repoPath string) *Client {
	return &Client{
		repoPath: repoPath,
	}
}

// Change represents a Jujutsu change
type Change struct {
	ID          string    `json:"change_id"`
	Description string    `json:"description"`
	Author      string    `json:"author"`
	Timestamp   time.Time `json:"timestamp"`
	Files       []string  `json:"files"`
}

// GetChangesSince returns all changes since the given change ID
func (c *Client) GetChangesSince(ctx context.Context, since string) ([]Change, error) {
	// Build jj log command
	args := []string{
		"log",
		"--no-graph",
		"--template", `{"change_id": "{change_id}", "description": "{description}", "author": "{author}", "timestamp": "{committer_timestamp}"}`,
	}

	if since != "" {
		args = append(args, fmt.Sprintf("%s..", since))
	}

	cmd := exec.CommandContext(ctx, "jj", args...)
	cmd.Dir = c.repoPath

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to run jj log: %v, stderr: %s", err, stderr.String())
	}

	// Parse output
	var changes []Change
	lines := strings.Split(stdout.String(), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		var change Change
		if err := json.Unmarshal([]byte(line), &change); err != nil {
			continue // Skip malformed lines
		}

		// Get files for this change
		files, err := c.getChangedFiles(ctx, change.ID)
		if err == nil {
			change.Files = files
		}

		changes = append(changes, change)
	}

	return changes, nil
}

// getChangedFiles returns files changed in a specific change
func (c *Client) getChangedFiles(ctx context.Context, changeID string) ([]string, error) {
	cmd := exec.CommandContext(ctx, "jj", "diff", "--summary", "-r", changeID)
	cmd.Dir = c.repoPath

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to get changed files: %v, stderr: %s", err, stderr.String())
	}

	// Parse file list
	var files []string
	lines := strings.Split(stdout.String(), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		// Extract filename from diff output (format: "M path/to/file")
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			files = append(files, parts[1])
		}
	}

	return files, nil
}

// GetFileContent returns the content of a file at a specific change
func (c *Client) GetFileContent(ctx context.Context, changeID, filePath string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "jj", "cat", "-r", changeID, filePath)
	cmd.Dir = c.repoPath

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to get file content: %v, stderr: %s", err, stderr.String())
	}

	return stdout.Bytes(), nil
}

// GetCurrentRevision returns the current change ID
func (c *Client) GetCurrentRevision(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, "jj", "log", "-r", "@", "--no-graph", "--template", "{change_id}")
	cmd.Dir = c.repoPath

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to get current revision: %v, stderr: %s", err, stderr.String())
	}

	return strings.TrimSpace(stdout.String()), nil
}

// Sync updates the local repository
func (c *Client) Sync(ctx context.Context) error {
	// Fetch from git-colocated repository if exists
	cmd := exec.CommandContext(ctx, "jj", "git", "fetch")
	cmd.Dir = c.repoPath

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		// It's OK if git fetch fails (might not be colocated)
		return nil
	}

	return nil
}

// IsKubernetesManifest checks if a file is a Kubernetes manifest
func IsKubernetesManifest(filePath string) bool {
	ext := strings.ToLower(filePath)
	return strings.HasSuffix(ext, ".yaml") || strings.HasSuffix(ext, ".yml")
}

// FilterKubernetesManifests filters a list of files for Kubernetes manifests
func FilterKubernetesManifests(files []string, basePath string) []string {
	var manifests []string
	for _, file := range files {
		// Check if file is under the base path
		if !strings.HasPrefix(file, basePath) {
			continue
		}

		// Check if it's a Kubernetes manifest
		if IsKubernetesManifest(file) {
			manifests = append(manifests, file)
		}
	}
	return manifests
}
