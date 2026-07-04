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
Set-Content -LiteralPath $stdoutPath -Value "" -Encoding UTF8
Set-Content -LiteralPath $stderrPath -Value "" -Encoding UTF8
Set-Content -LiteralPath $finalMessagePath -Value "" -Encoding UTF8

$promptText = Get-Content -LiteralPath $promptPath -Raw

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
try {
    if ($resolvedNodeDirectory) {
        $env:PATH = "$resolvedNodeDirectory;$previousPath"
    }
    Set-Location -LiteralPath $repoPath
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
    if ($hadNativePreference) {
        $PSNativeCommandUseErrorActionPreference = $previousNativePreference
    }
}

$stdout = Get-Content -LiteralPath $stdoutPath -Raw
$stderr = Get-Content -LiteralPath $stderrPath -Raw

Set-Content -LiteralPath $stdoutPath -Value $stdout -Encoding UTF8
Set-Content -LiteralPath $stderrPath -Value $stderr -Encoding UTF8

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

Set-Content -LiteralPath $finalMessagePath -Value $finalMessage -Encoding UTF8

exit $exitCode
