param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$PythonCommand = "py -3.12"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$pythonParts = $PythonCommand -split " "
$pythonExe = $pythonParts[0]
$pythonArgs = @()
if ($pythonParts.Length -gt 1) {
    $pythonArgs = $pythonParts[1..($pythonParts.Length - 1)]
}

& $pythonExe @pythonArgs -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 is required."
}

& $pythonExe @pythonArgs -m venv .venv
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -c constraints/ci-py312.txt -e ".[core,dev]"
& $venvPython scripts/generate_config_schema.py
& $venvPython scripts/generate_repository_manifest.py
& $venvPython scripts/verify_repository.py
& $venvPython -m mapel_linkage init-local-project --directory .
& $venvPython -m mapel_linkage doctor --project-root .
& $venvPython -m pytest -q tests/end_to_end/test_complete_synthetic_vertical_slice.py

Write-Output "Local Linkage Engine workspace initialized and synthetic smoke test passed."
