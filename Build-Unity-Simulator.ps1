param([string]$UnityEditorPath)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'local-results') -Force | Out-Null
. (Join-Path $PSScriptRoot 'Resolve-UnityEditor.ps1')
$unityPath = Resolve-UnityEditor $UnityEditorPath
$projectPath = Join-Path $PSScriptRoot 'UnitySimulator'
$logPath = Join-Path $PSScriptRoot 'local-results\unity-build.log'
# A cached Hub login is used by the licensing client; start Hub if it is closed.
if (-not (Get-Process -Name 'Unity Hub' -ErrorAction SilentlyContinue)) {
    $hubPackage = Get-AppxPackage -Name '*UnityHub*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hubPackage) {
        $hubExecutable = Join-Path $hubPackage.InstallLocation 'app\Unity Hub.exe'
        if (Test-Path -LiteralPath $hubExecutable) {
            Start-Process -FilePath $hubExecutable -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
    }
}
for ($attempt = 1; $attempt -le 3; $attempt++) {
    $buildProcess = Start-Process -FilePath $unityPath -ArgumentList @('-batchmode','-quit','-projectPath',('"'+$projectPath+'"'),'-executeMethod','BuildSimulator.Build','-logFile',('"'+$logPath+'"')) -WindowStyle Hidden -Wait -PassThru
    if ($buildProcess.ExitCode -eq 0) { break }
    if ($buildProcess.ExitCode -ne 198 -or $attempt -eq 3) { throw "Unity build failed ($($buildProcess.ExitCode)). See $logPath. For code 198, check Hub license-client connection; account login alone does not identify the cause." }
    Copy-Item -LiteralPath $logPath -Destination (Join-Path $PSScriptRoot "local-results\unity-license-retry-$attempt.log") -Force
    Write-Host "Unity license-client connection not ready. Retrying ($attempt/3)..."
    Start-Sleep -Seconds 3
}
if (-not (Test-Path 'UnityBuild\AimAndShoot.exe')) { throw 'Unity did not produce an executable. Check the build log.' }
Write-Host 'Build complete. Run Start-Unity-Simulator.cmd.'

