"""Git history API for viewing file versions."""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/git")


class GitCommit(BaseModel):
    """A git commit entry."""
    hash: str           # Short hash (7 chars)
    full_hash: str      # Full hash
    message: str        # Commit message (first line)
    author: str         # Author name
    date: str           # ISO format date
    relative_date: str  # e.g., "2 days ago"


class GitHistoryResponse(BaseModel):
    """Response for git history."""
    success: bool
    file_path: str
    commits: list[GitCommit]
    current_status: str  # "clean", "modified", "untracked"
    error: Optional[str] = None


class GitFileResponse(BaseModel):
    """Response for file at specific commit."""
    success: bool
    commit: str
    file_path: str
    content: str
    error: Optional[str] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is at web_helper/backend/api/git.py
    return Path(__file__).parent.parent.parent.parent


def run_git_command(args: list[str], cwd: Path) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, str(e)


def get_agent_file_path(agent_name: str) -> Path:
    """Get the file path for an agent."""
    root = get_project_root()
    # Remove .py extension if present
    agent_name = agent_name.replace(".py", "")
    return root / "cvlabkit" / "agent" / f"{agent_name}.py"


@router.get("/history/{agent_name}")
async def get_git_history(
    agent_name: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max commits to return"),
) -> GitHistoryResponse:
    """Get git commit history for an agent file.

    Returns list of commits that modified this file, with most recent first.
    """
    root = get_project_root()
    file_path = get_agent_file_path(agent_name)
    relative_path = file_path.relative_to(root)

    # Check if file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Agent file not found: {agent_name}")

    # Get current status
    success, status_output = run_git_command(
        ["status", "--porcelain", str(relative_path)],
        root
    )

    if not success:
        current_status = "unknown"
    elif status_output == "":
        current_status = "clean"
    elif status_output.startswith("??"):
        current_status = "untracked"
    else:
        current_status = "modified"

    # Get commit history
    # Format: hash|full_hash|message|author|date|relative_date
    format_str = "%h|%H|%s|%an|%aI|%ar"
    success, output = run_git_command(
        ["log", f"--pretty=format:{format_str}", f"-n{limit}", "--", str(relative_path)],
        root
    )

    if not success:
        return GitHistoryResponse(
            success=False,
            file_path=str(relative_path),
            commits=[],
            current_status=current_status,
            error=output,
        )

    commits = []
    if output:
        for line in output.split("\n"):
            parts = line.split("|")
            if len(parts) >= 6:
                commits.append(GitCommit(
                    hash=parts[0],
                    full_hash=parts[1],
                    message=parts[2],
                    author=parts[3],
                    date=parts[4],
                    relative_date=parts[5],
                ))

    return GitHistoryResponse(
        success=True,
        file_path=str(relative_path),
        commits=commits,
        current_status=current_status,
    )


@router.get("/file/{commit}/{agent_name}")
async def get_file_at_commit(
    commit: str,
    agent_name: str,
) -> GitFileResponse:
    """Get the content of an agent file at a specific commit.

    Use commit="HEAD" for the latest committed version.
    Use commit="working" for the current working directory version.
    """
    root = get_project_root()
    file_path = get_agent_file_path(agent_name)
    relative_path = file_path.relative_to(root)

    # Special case: "working" means current file
    if commit == "working":
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Agent file not found: {agent_name}")

        try:
            content = file_path.read_text(encoding="utf-8")
            return GitFileResponse(
                success=True,
                commit="working",
                file_path=str(relative_path),
                content=content,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    # Get file content at specific commit
    success, output = run_git_command(
        ["show", f"{commit}:{relative_path}"],
        root
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Could not get file at commit {commit}: {output}"
        )

    return GitFileResponse(
        success=True,
        commit=commit,
        file_path=str(relative_path),
        content=output,
    )
