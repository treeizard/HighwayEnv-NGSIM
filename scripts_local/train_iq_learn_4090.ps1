param(
    [string]$Python = "",
    [string]$CondaEnv = "ngsim_env",
    [string]$ExpertData = "expert_data/ngsim_ps_unified_expert_continuous_55145982",
    [string]$RunName = "",
    [int]$TotalUpdates = 60000,
    [int]$EvalEvery = 1000,
    [int]$EvalEpisodes = 4,
    [int]$BatchSize = 8192,
    [int]$ReplaySize = 200000,
    [int]$MaxExpertSamples = 200000,
    [int]$HiddenSize = 256,
    [int]$WorkerThreads = 4,
    [int]$CheckpointEvery = 10000,
    [int]$CheckpointVideoSteps = 120,
    [double]$LearningRate = 1e-4,
    [double]$QLearningRate = 1e-4,
    [double]$Gamma = 0.95,
    [double]$IqAlpha = 0.05,
    [double]$TargetTau = 0.002,
    [double]$BcCoef = 0.5,
    [double]$BcWarmupCoef = 2.0,
    [int]$BcWarmupUpdates = 5000,
    [int]$PolicyBcOnlyUpdates = 1000,
    [double]$QL2Coef = 0.001,
    [double]$QPolicyRegCoef = 0.001,
    [double]$ConservativeQCoef = 0.05,
    [double]$TargetEvalMeanLength = 190.0,
    [double]$TargetBcLoss = 0.03,
    [double]$MaxEvalCrashes = 0.0,
    [double]$MaxQAbs = 100.0,
    [switch]$AbortOnStalledConvergence,
    [switch]$SaveVideo,
    [switch]$OnlineWandb
)

$ErrorActionPreference = "Stop"

$RepoDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoDir

if ([string]::IsNullOrWhiteSpace($Python)) {
    $CondaBase = (& conda info --base).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($CondaBase)) {
        throw "Could not resolve conda base. Pass -Python explicitly, for example -Python 'D:\Coding Applications\miniconda\envs\ngsim_env\python.exe'."
    }
    $Python = Join-Path $CondaBase ("envs/{0}/python.exe" -f $CondaEnv)
}
if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "iqlearn_4090_local_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$env:PYTHONPATH = "{0}{1}{2}" -f $RepoDir, [IO.Path]::PathSeparator, $env:PYTHONPATH
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "$WorkerThreads"
$env:MKL_NUM_THREADS = "$WorkerThreads"
$env:OPENBLAS_NUM_THREADS = "$WorkerThreads"
$env:NUMEXPR_NUM_THREADS = "$WorkerThreads"
$IsWindowsOS = [System.Environment]::OSVersion.Platform -eq "Win32NT"
if (-not $IsWindowsOS) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

New-Item -ItemType Directory -Force -Path "logs", "logs/iq_learn" | Out-Null

Write-Host "Repository: $RepoDir"
Write-Host "Python: $Python"
Write-Host "Expert data: $ExpertData"
Write-Host "Run name: $RunName"
Write-Host "GPU target: CUDA device 0"
Write-Host "Throughput: batch=$BatchSize replay=$ReplaySize max_expert=$MaxExpertSamples"

& $Python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) {
    throw "Python/Torch CUDA check failed."
}

$wandbMode = if ($OnlineWandb) { "online" } else { "disabled" }
$videoArg = if ($SaveVideo) { "--save-checkpoint-video" } else { "--no-save-checkpoint-video" }
$stalledArg = if ($AbortOnStalledConvergence) { "--abort-on-stalled-convergence" } else { "--no-abort-on-stalled-convergence" }

$TrainArgs = @(
    "scripts_gail/train_simple_iq_learn.py",
    "--expert-data", $ExpertData,
    "--scene", "us-101",
    "--action-mode", "continuous",
    "--episode-root", "highway_env/data/processed_20s",
    "--prebuilt-split", "train",
    "--device", "cuda",
    "--replay-device", "cuda",
    "--pin-cpu-replay",
    "--torch-matmul-precision", "high",
    "--total-updates", "$TotalUpdates",
    "--eval-every", "$EvalEvery",
    "--eval-episodes", "$EvalEpisodes",
    "--batch-size", "$BatchSize",
    "--replay-size", "$ReplaySize",
    "--max-expert-samples", "$MaxExpertSamples",
    "--hidden-size", "$HiddenSize",
    "--learning-rate", "$LearningRate",
    "--disc-learning-rate", "$QLearningRate",
    "--gamma", "$Gamma",
    "--iq-alpha", "$IqAlpha",
    "--target-tau", "$TargetTau",
    "--bc-coef", "$BcCoef",
    "--bc-warmup-coef", "$BcWarmupCoef",
    "--bc-warmup-updates", "$BcWarmupUpdates",
    "--policy-bc-only-updates", "$PolicyBcOnlyUpdates",
    "--q-l2-coef", "$QL2Coef",
    "--q-policy-reg-coef", "$QPolicyRegCoef",
    "--conservative-q-coef", "$ConservativeQCoef",
    "--target-value-clip", "20.0",
    "--policy-q-clip", "20.0",
    "--max-q-abs", "$MaxQAbs",
    "--save-best-checkpoint",
    "--abort-on-nonfinite",
    $stalledArg,
    "--convergence-crash-penalty", "25.0",
    "--target-eval-mean-length", "$TargetEvalMeanLength",
    "--target-bc-loss", "$TargetBcLoss",
    "--max-eval-crashes", "$MaxEvalCrashes",
    "--checkpoint-every", "$CheckpointEvery",
    $videoArg,
    "--checkpoint-video-steps", "$CheckpointVideoSteps",
    "--wandb-mode", $wandbMode,
    "--wandb-project", "highwayenv-ps-gail",
    "--wandb-tags", "iq-learn,continuous,local-4090,convergence-gated",
    "--run-name", $RunName
)

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "IQ-Learn training failed with exit code $LASTEXITCODE."
}

Write-Host "Done. Check logs/iq_learn/$RunName for final.pt and best.pt."
