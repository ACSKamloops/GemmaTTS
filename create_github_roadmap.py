#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=False, check=check)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def gh_exists_issue(repo: str, title: str) -> bool:
    try:
        out = capture(["gh", "issue", "list", "--repo", repo, "--search", f'in:title "{title}"', "--json", "title", "--limit", "100"])
        items = json.loads(out or "[]")
        return any(item.get("title") == title for item in items)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None, help="OWNER/REPO")
    parser.add_argument("--project-owner", default=None, help="GitHub login/org for the project. Defaults to repo owner.")
    parser.add_argument("--skip-project", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = json.loads((root / "roadmap-data.json").read_text(encoding="utf-8"))
    repo = args.repo or data["repo"]
    owner = args.project_owner or repo.split("/")[0]
    project_title = data["project_title"]

    for name, color in data["labels"]:
        run(["gh", "label", "create", name, "--repo", repo, "--color", color, "--force"], check=False)

    existing_ms = capture(["gh", "api", f"repos/{repo}/milestones", "--jq", ".[].title"]) if True else ""
    existing = set(existing_ms.splitlines())
    for title, desc in data["milestones"]:
        if title not in existing:
            run(["gh", "api", f"repos/{repo}/milestones", "-f", f"title={title}", "-f", f"description={desc}"], check=False)

    project_number = None
    if not args.skip_project:
        projects_json = capture(["gh", "project", "list", "--owner", owner, "--format", "json", "--limit", "100"])
        projects = json.loads(projects_json or "{}").get("projects", [])
        match = next((p for p in projects if p.get("title") == project_title), None)
        if not match:
            run(["gh", "project", "create", "--owner", owner, "--title", project_title])
            projects_json = capture(["gh", "project", "list", "--owner", owner, "--format", "json", "--limit", "100"])
            projects = json.loads(projects_json or "{}").get("projects", [])
            match = next((p for p in projects if p.get("title") == project_title), None)
        if match:
            project_number = str(match["number"])

    for issue in data["issues"]:
        title = issue["title"]
        if gh_exists_issue(repo, title):
            print(f"skip existing issue: {title}")
            continue
        body_path = root / issue["body_file"]
        cmd = [
            "gh", "issue", "create",
            "--repo", repo,
            "--title", title,
            "--body-file", str(body_path),
            "--milestone", issue["milestone"],
        ]
        for label in issue["labels"]:
            cmd.extend(["--label", label])
        if project_number:
            cmd.extend(["--project", project_title])
        run(cmd, check=False)

    print("done")


if __name__ == "__main__":
    main()
