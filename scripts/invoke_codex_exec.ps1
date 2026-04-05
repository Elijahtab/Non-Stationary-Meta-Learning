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
    [string]$Sandbox = "workspace-write",
    [string]$ApprovalMode = "never",
    [string]$Model = "",
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

$promptText = Get-Content -LiteralPath $promptPath -Raw

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.Add("exec")
$arguments.Add("-C")
$arguments.Add($repoPath)
$arguments.Add("-a")
$arguments.Add($ApprovalMode)
$arguments.Add("-s")
$arguments.Add($Sandbox)
$arguments.Add("--json")
$arguments.Add("-o")
$arguments.Add($finalMessagePath)
if ($Ephemeral) {
    $arguments.Add("--ephemeral")
}
if ($Model) {
    $arguments.Add("-m")
    $arguments.Add($Model)
}
$arguments.Add("-")

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $CodexBinary
$startInfo.WorkingDirectory = $repoPath
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true

foreach ($arg in $arguments) {
    [void]$startInfo.ArgumentList.Add($arg)
}

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo

if (-not $process.Start()) {
    throw "Failed to start Codex process."
}

$process.StandardInput.Write($promptText)
$process.StandardInput.Close()

$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()

Set-Content -LiteralPath $stdoutPath -Value $stdout -Encoding UTF8
Set-Content -LiteralPath $stderrPath -Value $stderr -Encoding UTF8

if ($stdout) {
    Write-Output $stdout
}
if ($stderr) {
    Write-Error $stderr
}

exit $process.ExitCode
