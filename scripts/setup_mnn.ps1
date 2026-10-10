# Set up the MNN export toolchain that Model-Conversion/convert_to_mnn.py uses (Windows):
#   1. MNN source at the pinned commit, in MNN\ at the repo root (git-ignored)
#   2. MNN\build\Release\MNNConvert.exe, the graph converter llmexport.py is handed via --mnnconvert
#   3. .venv_mnn at the repo root, llmexport.py's own Python environment (requirements-mnn.txt)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_mnn.ps1          # skips stages already done
#   powershell -ExecutionPolicy Bypass -File scripts\setup_mnn.ps1 -Force   # rebuilds and recreates
#
# Needs git, cmake >= 3.22, Visual Studio 2022 (or its Build Tools) with "Desktop development
# with C++", and Python 3.10-3.12 via the py launcher (override with -Python <path>).
param(
    [switch]$Force,
    [string]$Python = ""
)
# Not "Stop": Windows PowerShell turns a native command's redirected stderr into a terminating
# error under it. Failures are caught by exit code in Invoke-Checked instead.
$ErrorActionPreference = "Continue"

$MnnCommit = "47ccf6c6bb5b6d357cd1f9b4370cbecb6188fd34"   # MNN 3.6.1, the version the reference pipeline used
$RepoRoot = Split-Path -Parent $PSScriptRoot
$MnnRoot = Join-Path $RepoRoot "MNN"
$Venv = Join-Path $RepoRoot ".venv_mnn"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Jobs = [Environment]::ProcessorCount

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Find-MnnConvert {
    foreach ($candidate in @("build\Release\MNNConvert.exe", "build\MNNConvert.exe")) {
        $path = Join-Path $MnnRoot $candidate
        if (Test-Path $path) { return $path }
    }
    return $null
}

# 1. MNN source at the pinned commit
if (-not (Test-Path (Join-Path $MnnRoot ".git"))) {
    Write-Host "[setup] fetching MNN @ $($MnnCommit.Substring(0, 12)) into $MnnRoot"
    Invoke-Checked git @("init", "-q", $MnnRoot)
    Invoke-Checked git @("-C", $MnnRoot, "remote", "add", "origin", "https://github.com/alibaba/MNN.git")
}
$head = (& git -C $MnnRoot rev-parse HEAD 2>$null)
if ($head -ne $MnnCommit) {
    Invoke-Checked git @("-C", $MnnRoot, "fetch", "--depth", "1", "origin", $MnnCommit)
    Invoke-Checked git @("-C", $MnnRoot, "checkout", "-q", "--detach", "FETCH_HEAD")
    $Force = $true
}
Write-Host "=== MNN @ $(& git -C $MnnRoot rev-parse --short=12 HEAD) ==="

# 2. MNNConvert
if ($Force) { Remove-Item -Recurse -Force (Join-Path $MnnRoot "build") -ErrorAction SilentlyContinue }
if (-not (Find-MnnConvert)) {
    Write-Host "[setup] building MNNConvert"
    Invoke-Checked cmake @("-S", $MnnRoot, "-B", (Join-Path $MnnRoot "build"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMNN_BUILD_CONVERTER=ON",
        "-DMNN_BUILD_LLM=ON",
        "-DMNN_LOW_MEMORY=ON",
        "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON")
    Invoke-Checked cmake @("--build", (Join-Path $MnnRoot "build"), "--config", "Release",
        "--target", "MNNConvert", "-j", "$Jobs")
}
$MnnConvert = Find-MnnConvert
if (-not $MnnConvert) { throw "MNNConvert did not build" }

# 3. .venv_mnn
if ($Force) { Remove-Item -Recurse -Force $Venv -ErrorAction SilentlyContinue }
$venvReady = $false
if (Test-Path $VenvPython) {
    & $VenvPython -c "import torch, onnx, onnxslim, transformers" 2>$null
    $venvReady = ($LASTEXITCODE -eq 0)
}
if (-not $venvReady) {
    Write-Host "[setup] creating $Venv"
    Remove-Item -Recurse -Force $Venv -ErrorAction SilentlyContinue
    if ($Python) {
        Invoke-Checked $Python @("-m", "venv", $Venv)
    } else {
        Invoke-Checked py @("-3.12", "-m", "venv", $Venv)
    }
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", (Join-Path $RepoRoot "requirements-mnn.txt"))
}
Invoke-Checked $VenvPython @("-c", "import torch, transformers, onnx; print(f'llmexport env: torch {torch.__version__}, transformers {transformers.__version__}, onnx {onnx.__version__}')")

Write-Host ""
Write-Host "MNN export toolchain ready."
Write-Host "  MNN         $MnnRoot"
Write-Host "  MNNConvert  $MnnConvert"
Write-Host "  exporter    $VenvPython $MnnRoot\transformers\llm\export\llmexport.py"
