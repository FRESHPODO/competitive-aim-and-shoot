function Resolve-UnityEditor([string]$RequestedPath) {
    $candidates = @($RequestedPath, $env:UNITY_EDITOR_PATH)
    $localPath = Join-Path $PSScriptRoot 'UnityEditor.local.txt'
    if (Test-Path -LiteralPath $localPath) { $candidates += (Get-Content -LiteralPath $localPath -Raw).Trim() }
    $candidates += (Join-Path $env:ProgramFiles 'Unity\Hub\Editor\6000.3.22f1\Editor\Unity.exe')
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    throw 'Unity Editor not found. Pass -UnityEditorPath, set UNITY_EDITOR_PATH, or put the Unity.exe path in UnityEditor.local.txt.'
}
