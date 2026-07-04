param(
    [string]$Manifest = "config/research_manifest.toml",
    [int]$MaxTrials = 5,
    [string]$RunName = ("loop_{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
)

# Canonical local launcher for the autoresearch loop with Claude (Fable) as the trial
# agent. Detach-friendly: writes all output to autoresearch/<RunName>.log and drops
# autoresearch/<RunName>.exitcode on completion so external monitors can watch for it.
# The cached baseline is reused automatically when the fingerprint is unchanged.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:PYTHONPATH = "src"
$env:AUTORESEARCH_AGENT_COMMAND = 'powershell -ExecutionPolicy Bypass -File scripts\invoke_claude_exec.ps1 -PromptFile "{prompt_file}" -RepoRoot "{repo_root}" -FinalMessageFile "{final_message_file}" -StdoutFile "{agent_stdout_file}" -StderrFile "{agent_stderr_file}" -Model claude-fable-5'

& .\myenv\Scripts\python.exe scripts\run_autoresearch.py `
  --manifest $Manifest `
  --program config/program_neuromod.md `
  --max-trials $MaxTrials `
  --research-command ".\myenv\Scripts\python.exe scripts\run_research_trial.py --program {program} --manifest {manifest} --trial {trial} --trial-dir {trial_dir} --repo-root {repo_root} --baseline-file {baseline_file}" `
  *> "autoresearch\$RunName.log"

"EXIT:$LASTEXITCODE" | Out-File -FilePath "autoresearch\$RunName.exitcode" -Encoding ascii
