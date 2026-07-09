import subprocess
from typing import Dict, Any, Tuple

def check_gcloud_auth() -> Tuple[bool, str]:
    """Checks if gcloud CLI is authenticated."""
    try:
        res = subprocess.run(
            ["gcloud", "auth", "list", "--format=json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode != 0:
            return False, "gcloud CLI command failed: " + res.stderr
        
        # Parse output
        auths = res.stdout.strip()
        if not auths or auths == "[]":
            return False, "No active gcloud credentials found. Please run 'gcloud auth login'."
            
        return True, "Active gcloud credentials verified."
    except FileNotFoundError:
        return False, "gcloud CLI tool is not installed on system PATH."
    except Exception as e:
        return False, f"Unexpected error checking gcloud CLI: {str(e)}"

def check_tpu_availability(project: str, zone: str, name: str) -> Tuple[bool, str]:
    """Queries TPU status using gcloud CLI (mocked if no credentials)."""
    try:
        # Check command
        cmd = ["gcloud", "compute", "tpus", "describe", name, "--zone", zone, "--project", project, "--format=json"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return True, f"TPU VM '{name}' verified on {zone}."
        else:
            return False, f"TPU VM '{name}' not accessible: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Error calling compute API: {str(e)}"
