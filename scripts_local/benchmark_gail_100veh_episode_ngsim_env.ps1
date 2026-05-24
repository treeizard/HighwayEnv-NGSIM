param(
    [string]$Python = "",
    [string]$CondaEnv = "ngsim_env",
    [string]$Scene = "us-101",
    [string]$EpisodeRoot = "highway_env/data/processed_20s",
    [ValidateSet("train", "val", "test")]
    [string]$Split = "train",
    [int]$Steps = 200,
    [double]$RequestedControlledVehicles = 100,
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cuda",
    [int]$WorkerThreads = 2,
    [string]$PolicyModel = "recurrent_transformer",
    [int]$HiddenSize = 256,
    [int]$TransformerLayers = 2,
    [int]$TransformerHeads = 4,
    [string]$Checkpoint = "",
    [string]$Output = "",
    [switch]$NoProgress
)

$ErrorActionPreference = "Stop"

$RepoDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoDir

if ([string]::IsNullOrWhiteSpace($Python)) {
    $CondaBase = (& conda info --base).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($CondaBase)) {
        throw "Could not resolve conda base. Pass -Python explicitly, for example -Python 'D:\miniconda\envs\ngsim_env\python.exe'."
    }
    $Python = Join-Path $CondaBase ("envs/{0}/python.exe" -f $CondaEnv)
}
if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = "logs/benchmarks/gail_100veh_episode_${stamp}.json"
}

$env:PYTHONPATH = "{0}{1}{2}" -f $RepoDir, [IO.Path]::PathSeparator, $env:PYTHONPATH
$env:OMP_NUM_THREADS = "$WorkerThreads"
$env:MKL_NUM_THREADS = "$WorkerThreads"
$env:OPENBLAS_NUM_THREADS = "$WorkerThreads"
$env:NUMEXPR_NUM_THREADS = "$WorkerThreads"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

New-Item -ItemType Directory -Force -Path "logs", "logs/benchmarks" | Out-Null

Write-Host "Repository: $RepoDir"
Write-Host "Python: $Python"
Write-Host "Benchmark: GAIL-style rollout, requested controlled vehicles=$RequestedControlledVehicles, steps=$Steps, worker_threads=$WorkerThreads"
Write-Host "Policy: $PolicyModel hidden=$HiddenSize layers=$TransformerLayers heads=$TransformerHeads device=$Device"
Write-Host "Output: $Output"

& $Python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) {
    throw "Python/Torch check failed."
}

$ArgsList = @(
    "scripts_env_test/benchmark_gail_100veh_episode.py",
    "--scene", $Scene,
    "--episode-root", $EpisodeRoot,
    "--split", $Split,
    "--steps", "$Steps",
    "--requested-controlled-vehicles", "$RequestedControlledVehicles",
    "--device", $Device,
    "--worker-threads", "$WorkerThreads",
    "--policy-model", $PolicyModel,
    "--hidden-size", "$HiddenSize",
    "--transformer-layers", "$TransformerLayers",
    "--transformer-heads", "$TransformerHeads",
    "--output", $Output
)

if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    $ArgsList += @("--checkpoint", $Checkpoint)
}
if ($NoProgress) {
    $ArgsList += @("--disable-progress")
}

& $Python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "GAIL 100-vehicle episode benchmark failed with exit code $LASTEXITCODE."
}
