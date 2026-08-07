#!/usr/bin/env python3
"""
CalVer Release & Publish Script for blks.

Calculates the next CalVer version (YYYY.M.BUILD), updates metadata files,
creates a git commit and tag, and pushes to GitHub to trigger PyPI publishing.
"""

from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

ROOT_DIR = Path(__file__).parent.parent.resolve()
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
INIT_PATH = ROOT_DIR / "src" / "blks" / "__init__.py"


def get_current_git_tags() -> list[str]:
    """Fetch all git tags from the repository."""
    try:
        res = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return [t.strip() for t in res.stdout.splitlines() if t.strip()]
    except Exception:
        return []


def calculate_next_version(year: int, month: int) -> str:
    """
    Calculate next CalVer version string for given year & month.
    If existing tags/version match YYYY.M.X, increments X.
    If it's a new month/year, starts at YYYY.M.1.
    """
    prefix = f"{year}.{month}."
    highest_build = 0

    # 1. Check existing git tags matching vYYYY.M.*
    tags = get_current_git_tags()
    for tag in tags:
        clean_tag = tag.lstrip("v")
        if clean_tag.startswith(prefix):
            try:
                build_num = int(clean_tag[len(prefix) :])
                highest_build = max(highest_build, build_num)
            except ValueError:
                pass

    # 2. Check pyproject.toml version
    if PYPROJECT_PATH.exists():
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"(.*?)"', content, re.MULTILINE)
        if match:
            curr_ver = match.group(1)
            if curr_ver.startswith(prefix):
                try:
                    build_num = int(curr_ver[len(prefix) :])
                    highest_build = max(highest_build, build_num)
                except ValueError:
                    pass

    next_build = highest_build + 1
    return f"{year}.{month}.{next_build}"


def update_pyproject(new_version: str) -> None:
    """Update version in pyproject.toml."""
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    new_content = re.sub(
        r'^version\s*=\s*".*?"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    PYPROJECT_PATH.write_text(new_content, encoding="utf-8")


def update_init(new_version: str) -> None:
    """Update __version__ in src/blks/__init__.py."""
    if INIT_PATH.exists():
        content = INIT_PATH.read_text(encoding="utf-8")
        new_content = re.sub(
            r'^__version__\s*=\s*".*?"',
            f'__version__ = "{new_version}"',
            content,
            flags=re.MULTILINE,
        )
        INIT_PATH.write_text(new_content, encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    """Run a shell command and print progress."""
    print(f"  → Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def main():
    now = datetime.now()
    year = now.year
    month = now.month

    next_ver = calculate_next_version(year, month)
    tag_name = f"v{next_ver}"

    print(f"\n🚀 Preparing release for blks version {next_ver}...")

    # Step 1: Update metadata files
    print("\n1. Updating version in pyproject.toml & __init__.py...")
    update_pyproject(next_ver)
    update_init(next_ver)

    # Step 2: Git commit & tag
    print("\n2. Creating git commit & tag...")
    run_cmd(["git", "add", "."])
    run_cmd(["git", "commit", "-m", f"release: {next_ver}"])
    run_cmd(["git", "tag", tag_name])

    # Step 3: Git push
    print("\n3. Pushing release to GitHub...")
    run_cmd(["git", "push"])
    run_cmd(["git", "push", "origin", tag_name])

    print(f"\n✨ Successfully pushed release {next_ver} ({tag_name})!")
    print("GitHub Actions is now publishing your release to PyPI:")
    print("  • GitHub Actions: https://github.com/mattkellymt/blks/actions")
    print("  • PyPI Package:   https://pypi.org/project/blks/\n")


if __name__ == "__main__":
    main()
