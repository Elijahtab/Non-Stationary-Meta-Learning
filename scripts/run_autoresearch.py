from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lifelong_learning.research.autoresearch import AutoresearchSupervisor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded autoresearch supervisor on the immutable benchmark stack."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="config/research_manifest.toml",
        help="Path to the autoresearch manifest.",
    )
    parser.add_argument(
        "--research-command",
        type=str,
        help=(
            "Shell command template to execute once per trial. "
            "Supports {trial}, {trial_dir}, {manifest}, {program}, {repo_root}, "
            "{session_id}, {session_dir}, and {baseline_file}."
        ),
    )
    parser.add_argument(
        "--program",
        type=str,
        help="Optional research brief or Markdown prompt path passed into the command template.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the frozen benchmark device from the manifest.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional override for the manifest max_trials limit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and print the resolved runtime config without running trials.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --validate-only with a session preview.",
    )
    args = parser.parse_args()

    supervisor = AutoresearchSupervisor(
        repo_root=REPO_ROOT,
        manifest_path=args.manifest,
        research_command=args.research_command or "",
        program_path=args.program,
        device=args.device,
    )

    if args.validate_only or args.dry_run:
        print(json.dumps(supervisor.run(dry_run=True), indent=2))
        return

    if not args.research_command:
        parser.error("--research-command is required unless --validate-only or --dry-run is used.")

    summary = supervisor.run(max_trials=args.max_trials)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
