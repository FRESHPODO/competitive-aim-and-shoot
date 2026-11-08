param([switch]$Matches,[switch]$Curriculum,[switch]$Generational,[switch]$RewardGenerational)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$config = Join-Path $PSScriptRoot 'python-path.txt'
$candidates = @()
if (Test-Path -LiteralPath $config) { $candidates += (Get-Content -LiteralPath $config -Raw -Encoding UTF8).Trim() }
$candidates += Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
foreach ($name in @('python.exe','python3.exe')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
}
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if ($py) { $candidates += (& $py.Source -3 -c 'import sys; print(sys.executable)' 2>$null) }
$check = 'import torch, stable_baselines3, numpy, pandas, scipy, matplotlib, yaml, psutil, joblib, tkinter, tqdm, seaborn, einops, tensorboard, filelock'
$selected = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $ErrorActionPreference = 'Continue'
        try { & $candidate -I -c $check 2>$null; $checkExit = $LASTEXITCODE } finally { $ErrorActionPreference = 'Stop' }
        if ($checkExit -eq 0) { $selected = $candidate; break }
    }
}
if (-not $selected) {
    Write-Host 'Enter python.exe path from an environment with PyTorch and replay dependencies installed.'
    $candidate = (Read-Host 'Python path').Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw 'Python executable not found.' }
    & $candidate -I -c $check
    if ($LASTEXITCODE -ne 0) { throw 'This Python environment is missing required dependencies.' }
    $selected = $candidate
}
Set-Content -LiteralPath $config -Value $selected -Encoding UTF8
$launchArgs = @('-I', (Join-Path $PSScriptRoot 'launch.py'))
if ($Matches) { $launchArgs += '--matches' }
if ($Curriculum) { $launchArgs += '--curriculum' }
if ($Generational) { $launchArgs += '--generational' }
if ($RewardGenerational) { $launchArgs += '--reward-generational' }
& $selected @launchArgs
exit $LASTEXITCODE
