from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lifelong_learning.research.autoresearch import load_research_manifest
from lifelong_learning.research.trial_prompt import (
    build_trial_context_payload,
    collect_trial_history,
    load_baseline_summary,
    load_hypothesis_queue,
    render_research_trial_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single autoresearch trial prompt and optionally invoke a research agent."
    )
    parser.add_argument("--program", type=str, required=True, help="Research brief Markdown file.")
    parser.add_argument("--manifest", type=str, required=True, help="Autoresearch manifest.")
    parser.add_argument("--trial", type=int, required=True, help="Trial index.")
    parser.add_argument("--trial-dir", type=str, required=True, help="Scratch directory for this trial.")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument(
        "--baseline-file",
        type=str,
        default=None,
        help="Optional baseline summary JSON emitted by the supervisor.",
    )
    parser.add_argument(
        "--runner-template",
        type=str,
        default=None,
        help=(
            "Optional shell command template for invoking the research agent. "
            "Can also be supplied through AUTORESEARCH_AGENT_COMMAND."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the prompt/context files and print the resolved runner command without executing it.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    program_path = Path(args.program).resolve()
    manifest_path = Path(args.manifest).resolve()
    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_research_manifest(manifest_path)
    program_text = program_path.read_text(encoding="utf-8")
    baseline_summary = load_baseline_summary(args.baseline_file)
    hypothesis_queue = load_hypothesis_queue(repo_root)
    trial_history = collect_trial_history(
        repo_root=repo_root,
        manifest=manifest,
        session_id=trial_dir.parent.name,
    )

    prompt = render_research_trial_prompt(
        program_text=program_text,
        trial_index=args.trial,
        repo_root=repo_root,
        manifest_path=manifest_path,
        program_path=program_path,
        trial_dir=trial_dir,
        manifest=manifest,
        baseline_summary=baseline_summary,
        hypothesis_queue=hypothesis_queue,
        trial_history=trial_history,
    )
    prompt_path = trial_dir / "research_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    context_payload = build_trial_context_payload(
        trial_index=args.trial,
        repo_root=repo_root,
        manifest_path=manifest_path,
        program_path=program_path,
        trial_dir=trial_dir,
        manifest=manifest,
        baseline_summary=baseline_summary,
        hypothesis_queue=hypothesis_queue,
        trial_history=trial_history,
    )
    context_path = trial_dir / "trial_context.json"
    context_path.write_text(json.dumps(context_payload, indent=2), encoding="utf-8")

    notes_path = trial_dir / "agent_notes.md"
    final_message_path = trial_dir / "codex_final_message.md"
    agent_stdout_path = trial_dir / "agent_runner.stdout.log"
    agent_stderr_path = trial_dir / "agent_runner.stderr.log"
    runner_template = args.runner_template or os.environ.get("AUTORESEARCH_AGENT_COMMAND")
    runner_command = None
    if runner_template:
        runner_command = runner_template.format(
            prompt_file=str(prompt_path),
            context_file=str(context_path),
            notes_file=str(notes_path),
            final_message_file=str(final_message_path),
            agent_stdout_file=str(agent_stdout_path),
            agent_stderr_file=str(agent_stderr_path),
            trial=args.trial,
            trial_dir=str(trial_dir),
            manifest=str(manifest_path),
            program=str(program_path),
            repo_root=str(repo_root),
            baseline_file=str(Path(args.baseline_file).resolve()) if args.baseline_file else "",
        )

    payload = {
        "trial": args.trial,
        "prompt_file": str(prompt_path),
        "context_file": str(context_path),
        "notes_file": str(notes_path),
        "final_message_file": str(final_message_path),
        "agent_stdout_file": str(agent_stdout_path),
        "agent_stderr_file": str(agent_stderr_path),
        "runner_command": runner_command,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    if not runner_command:
        raise SystemExit(
            "No AUTORESEARCH_AGENT_COMMAND or --runner-template configured. "
            "Use --dry-run to preview the prompt and wire in a runner first."
        )

    completed = subprocess.run(
        runner_command,
        cwd=str(repo_root),
        shell=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
