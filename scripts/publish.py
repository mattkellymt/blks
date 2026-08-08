#!/usr/bin/env python3
"""
CalVer Release & Publishing Script for blks.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).parent.parent.resolve()
WORKFLOW_NAME = "Publish to PyPI"


def get_version() -> str:
    """Read the current project version via `uv version`."""
    result = subprocess.run(
        ["uv", "version", "--short"],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def set_version(new_ver: str) -> None:
    """Set the project version, updating pyproject.toml and re-locking uv.lock."""
    subprocess.run(["uv", "version", new_ver], cwd=ROOT_DIR, check=True)


def update_version(current_version: str) -> str:
    """Calculate next YYYY.M.BUILD version based on today's date."""
    now = datetime.now()
    year = now.year
    month = now.month

    parts = current_version.split(".")
    if len(parts) == 3 and parts[0] == str(year) and parts[1] == str(month):
        next_build = int(parts[2]) + 1
    else:
        next_build = 1

    return f"{year}.{month}.{next_build}"


def wait_for_publish(tag_name: str) -> None:
    """Block until the tag-triggered publish workflow finishes, failing if it does not succeed.

    The tag push publishes to PyPI via GitHub Actions, so pushing the tag alone
    does not prove the release landed. This finds the run for the tag, then
    `gh run watch --exit-status` blocks until it completes and exits non-zero on failure.
    """
    print("Waiting for the publish workflow to start...")
    run_id = ""
    for _ in range(40):  # ~2 min, in case Actions is slow to register the run
        result = subprocess.run(
            ["gh", "run", "list", "--workflow", WORKFLOW_NAME,
             "--branch", tag_name, "--limit", "1",
             "--json", "databaseId", "--jq", ".[0].databaseId"],
            cwd=ROOT_DIR, capture_output=True, text=True, check=True,
        )
        run_id = result.stdout.strip()
        if run_id:
            break
        time.sleep(3)

    if not run_id:
        sys.exit(
            f"Error: no '{WORKFLOW_NAME}' run found for tag {tag_name}. "
            "The tag was pushed; check GitHub Actions manually."
        )

    print(f"Watching workflow run {run_id}...")
    # --exit-status returns non-zero (raising CalledProcessError via check=True) if the run fails.
    subprocess.run(
        ["gh", "run", "watch", run_id, "--exit-status"],
        cwd=ROOT_DIR, check=True,
    )


def main():
    current_ver = get_version()
    new_ver = update_version(current_ver)
    tag_name = f"v{new_ver}"

    print(f"Bumping version: {current_ver} -> {new_ver}")

    # 1. Update version (pyproject.toml + uv.lock)
    set_version(new_ver)

    # 2. Commit, tag, and push to GitHub
    subprocess.run(["git", "add", "."], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"release: {new_ver}"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "tag", tag_name], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push"], cwd=ROOT_DIR, check=True)
    subprocess.run(["git", "push", "origin", tag_name], cwd=ROOT_DIR, check=True)

    # 3. Wait for CI to actually publish before declaring success.
    wait_for_publish(tag_name)

    print(f"Released {tag_name} to GitHub & PyPI!")


if __name__ == "__main__":
    main()
