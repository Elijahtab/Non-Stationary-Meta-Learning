param(
    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$FinalMessageFile,

    [Parameter(Mandatory = $true)]
    [string]$StdoutFile,

    [Parameter(Mandatory = $true)]
    [string]$StderrFile,

    [string]$ClaudeBinary = "claude",
    [string]$Model = "claude-fable-5",
    [switch]$SkipPermissions = $true
)

# Non-interactive Claude Code (Fable) launcher for one autoresearch trial. Analogous to
# invoke_codex_exec.ps1: reads the rendered prompt, pipes it into `claude -p`, sets the repo
# workspace, uses unattended permissions, and writes stdout/stderr + the final assistant message
# to disk. The outer autoresearch supervisor (run_autoresearch.py) is the real safety boundary —
# it audits the resulting diff against the editable surface and rolls back violations.

$ErrorActionPreference = "Stop"

$promptPath = (Resolve-Path $PromptFile).Path
$repoPath = (Resolve-Path $RepoRoot).Path
$finalMessagePath = [System.IO.Path]::GetFullPath($FinalMessageFile)
$stdoutPath = [System.IO.Path]::GetFullPath($StdoutFile)
$stderrPath = [System.IO.Path]::GetFullPath($StderrFile)

$null = New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($finalMessagePath))
$null = New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($stdoutPath))
$null = New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($stderrPath))

# BOM-less UTF-8 everywhere: PS 5.1's Set-Content -Encoding UTF8 prepends a BOM, which
# breaks byte-exact/json consumers of these files (review 2026-07-03).
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($stdoutPath, "", $utf8NoBom)
[System.IO.File]::WriteAllText($stderrPath, "", $utf8NoBom)
[System.IO.File]::WriteAllText($finalMessagePath, "", $utf8NoBom)

# The prompt file is BOM-less UTF-8 written by Python; PS 5.1 would misread it as ANSI.
$promptText = [System.IO.File]::ReadAllText($promptPath, [System.Text.Encoding]::UTF8)

function Resolve-ClaudeCommand {
    param([string]$Requested)

    $candidates = [System.Collections.Generic.List[string]]::new()
    $hasExtension = -not [string]::IsNullOrEmpty([System.IO.Path]::GetExtension($Requested))
    $appDataNpm = Join-Path $env:APPDATA "npm"

    if (-not $hasExtension) {
        foreach ($candidate in @(
            (Join-Path $appDataNpm "$Requested.ps1"),
            (Join-Path $appDataNpm "$Requested.cmd"),
            (Join-Path $appDataNpm "$Requested.exe"),
            "$Requested.ps1",
            "$Requested.cmd",
            "$Requested.exe",
            $Requested
        )) {
            if (-not [string]::IsNullOrWhiteSpace($candidate)) {
                [void]$candidates.Add($candidate)
            }
        }
    }
    else {
        [void]$candidates.Add($Requested)
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }

        $commands = @(Get-Command $candidate -All -ErrorAction SilentlyContinue)
        foreach ($command in $commands) {
            if ($command.Source) {
                return $command.Source
            }
        }
    }

    if (Test-Path $Requested) {
        return (Resolve-Path $Requested).Path
    }

    throw "Unable to resolve Claude command '$Requested'."
}

function Resolve-NodeDirectory {
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($nodeCommand -and $nodeCommand.Source) {
        return [System.IO.Path]::GetDirectoryName($nodeCommand.Source)
    }

    foreach ($candidate in @(
        "C:\Program Files\nodejs\node.exe",
        "C:\Program Files (x86)\nodejs\node.exe"
    )) {
        if (Test-Path $candidate) {
            return [System.IO.Path]::GetDirectoryName($candidate)
        }
    }

    return $null
}

$execArguments = [System.Collections.Generic.List[string]]::new()
$execArguments.Add("-p")
$execArguments.Add("--add-dir")
$execArguments.Add($repoPath)
$execArguments.Add("--output-format")
$execArguments.Add("json")
if ($SkipPermissions) {
    $execArguments.Add("--dangerously-skip-permissions")
}
if ($Model) {
    $execArguments.Add("--model")
    $execArguments.Add($Model)
}

$resolvedClaude = Resolve-ClaudeCommand $ClaudeBinary
$resolvedNodeDirectory = Resolve-NodeDirectory
$previousPath = $env:PATH
$previousLocation = Get-Location
$previousErrorActionPreference = $ErrorActionPreference
$hadNativePreference = $false
$previousNativePreference = $null
$previousOutputEncoding = $OutputEncoding
try {
    if ($resolvedNodeDirectory) {
        $env:PATH = "$resolvedNodeDirectory;$previousPath"
    }
    Set-Location -LiteralPath $repoPath
    # PS 5.1 pipes to native processes using $OutputEncoding (default US-ASCII), which
    # irreversibly turns every non-ASCII prompt character into '?' (review 2026-07-03).
    # Known residue: PS 5.1's pipe writer still emits one UTF-8 preamble regardless of the
    # BOM-less instance, so the CLI sees a single leading U+FEFF — harmless in a prompt
    # (verified empirically 2026-07-03; the .sh twin is byte-exact via `< file`).
    $OutputEncoding = $utf8NoBom
    $ErrorActionPreference = "Continue"
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $hadNativePreference = $true
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $promptText | & $resolvedClaude @execArguments 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $previousPath
    Set-Location $previousLocation
    $ErrorActionPreference = $previousErrorActionPreference
    $OutputEncoding = $previousOutputEncoding
    if ($hadNativePreference) {
        $PSNativeCommandUseErrorActionPreference = $previousNativePreference
    }
}

# A binary that resolved but failed to launch leaves $LASTEXITCODE untouched ($null or a
# stale value from an earlier command) — `exit $null` would report SUCCESS for an agent
# run that never happened (review 2026-07-03, reproduced). Treat unknown as failure.
if ($null -eq $exitCode) {
    $exitCode = 1
}

$stdout = Get-Content -LiteralPath $stdoutPath -Raw
$stderr = Get-Content -LiteralPath $stderrPath -Raw
if ($null -eq $stdout) { $stdout = "" }
if ($null -eq $stderr) { $stderr = "" }

[System.IO.File]::WriteAllText($stdoutPath, $stdout, $utf8NoBom)
[System.IO.File]::WriteAllText($stderrPath, $stderr, $utf8NoBom)

# Parse the machine-readable result and extract the final assistant message so downstream
# tooling can read it the same way it reads Codex's `-o` final-message file. Fall back to raw
# stdout if the JSON is unparseable, and surface an agent-reported error as a non-zero exit.
$finalMessage = $stdout
if (-not [string]::IsNullOrWhiteSpace($stdout)) {
    try {
        $parsed = $stdout | ConvertFrom-Json
        if ($null -ne $parsed.result) {
            $finalMessage = $parsed.result
        }
        if ($parsed.is_error -eq $true -and $exitCode -eq 0) {
            $exitCode = 1
        }
    }
    catch {
        # Leave $finalMessage as raw stdout; the JSON was not well-formed.
    }
}

[System.IO.File]::WriteAllText($finalMessagePath, [string]$finalMessage, $utf8NoBom)

exit $exitCode
