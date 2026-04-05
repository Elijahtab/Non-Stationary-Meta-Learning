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

[string]$CodexBinary = "codex",
    [string]$Sandbox = "danger-full-access",
    [string]$ApprovalMode = "never",
    [string]$Model = "gpt-5.4-mini",
    [switch]$Ephemeral
)

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

$promptText = Get-Content -LiteralPath $promptPath -Raw

function Resolve-CodexCommand {
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

    throw "Unable to resolve Codex command '$Requested'."
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

$globalArguments = [System.Collections.Generic.List[string]]::new()
$globalArguments.Add("-a")
$globalArguments.Add($ApprovalMode)

$execArguments = [System.Collections.Generic.List[string]]::new()
$execArguments.Add("exec")
$execArguments.Add("-C")
$execArguments.Add($repoPath)
$execArguments.Add("-s")
$execArguments.Add($Sandbox)
$execArguments.Add("--json")
$execArguments.Add("-o")
$execArguments.Add($finalMessagePath)
if ($Ephemeral) {
    $execArguments.Add("--ephemeral")
}
if ($Model) {
    $execArguments.Add("-m")
    $execArguments.Add($Model)
}
$execArguments.Add("-")

$resolvedCodex = Resolve-CodexCommand $CodexBinary
$resolvedNodeDirectory = Resolve-NodeDirectory
$previousPath = $env:PATH
$previousErrorActionPreference = $ErrorActionPreference
$hadNativePreference = $false
$previousNativePreference = $null
try {
    if ($resolvedNodeDirectory) {
        $env:PATH = "$resolvedNodeDirectory;$previousPath"
    }
    $ErrorActionPreference = "Continue"
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $hadNativePreference = $true
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $promptText | & $resolvedCodex @globalArguments @execArguments 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $previousPath
    $ErrorActionPreference = $previousErrorActionPreference
    if ($hadNativePreference) {
        $PSNativeCommandUseErrorActionPreference = $previousNativePreference
    }
}

$stdout = Get-Content -LiteralPath $stdoutPath -Raw
$stderr = Get-Content -LiteralPath $stderrPath -Raw

Set-Content -LiteralPath $stdoutPath -Value $stdout -Encoding UTF8
Set-Content -LiteralPath $stderrPath -Value $stderr -Encoding UTF8

exit $exitCode
