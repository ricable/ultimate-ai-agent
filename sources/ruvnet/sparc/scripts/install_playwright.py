from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import subprocess
import sys

class PlaywrightInstallHook(BuildHookInterface):
    def initialize(self, version, build_data):
        try:
            subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install playwright browsers: {e}", file=sys.stderr)
