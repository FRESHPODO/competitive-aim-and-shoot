param([string]$UnityEditorPath)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'local-results') -Force | Out-Null
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python environment missing. Follow README.md installation first.' }
. (Join-Path $PSScriptRoot 'Resolve-UnityEditor.ps1')
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 2
    if ($health.model -ne 'ijhcs_default') { throw 'Port 8765 is in use by another application.' }
} catch {
    Start-Process -FilePath $pythonPath -ArgumentList '-m unity_bridge.server' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput 'local-results\bridge-output.log' -RedirectStandardError 'local-results\bridge-error.log'
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try { $health = Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 1; $ready = $health.model -eq 'ijhcs_default'; if ($ready) { break } } catch {}
    }
    if (-not $ready) { throw 'AI bridge did not start. See local-results\bridge-error.log.' }
}
$appPath = Join-Path $PSScriptRoot 'UnityBuild\AimAndShoot.exe'
if (Test-Path -LiteralPath $appPath) {
    Start-Process -FilePath $appPath -WorkingDirectory $PSScriptRoot
} else {
    Write-Host 'Opening Unity project. Use Tools > Aim and Shoot > Create scene, then Play.'
    $unityPath = Resolve-UnityEditor $UnityEditorPath
    $projectPath = Join-Path $PSScriptRoot 'UnitySimulator'
    Start-Process -FilePath $unityPath -ArgumentList @('-projectPath', ('"' + $projectPath + '"'))
}
