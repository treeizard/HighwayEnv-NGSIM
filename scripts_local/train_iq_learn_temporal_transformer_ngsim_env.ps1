param(
    [string]$Python = "",
    [string]$CondaEnv = "ngsim_env",
    [string]$ExpertData = "expert_data/ngsim_ps_unified_expert_continuous_55145982",
    [string]$RunName = "",
    [string]$Scene = "us-101",
    [string]$EpisodeRoot = "highway_env/data/processed_20s",
    [string]$TrainSplit = "train",
    [string]$ValidationSplit = "val",
    [string]$TestSplit = "test",
    [int]$TotalUpdates = 60000,
    [int]$EvalEvery = 1000,
    [int]$ValidationEvery = 3000,
    [int]$ValidationEpisodes = 4,
    [int]$TestEpisodes = 4,
    [int]$BatchSize = 32,
    [int]$ReplaySize = 200000,
    [int]$MaxExpertSamples = 200000,
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$ReplayDevice = "cpu",
    [int]$HiddenSize = 256,
    [int]$TransformerLayers = 2,
    [int]$TransformerHeads = 4,
    [double]$TransformerDropout = 0.1,
    [int]$TransformerTemporalKernelSize = 5,
    [int]$TransformerTemporalLayers = 1,
    [int]$TransformerMemoryTokens = 8,
    [int]$TransformerMemoryContextLength = 32,
    [int]$TransformerRecurrentSequenceLength = 32,
    [int]$TransformerRecurrentSequencesPerBatch = 32,
    [int]$TransformerRecurrentMicroBatchSequences = 8,
    [int]$WorkerThreads = 4,
    [int]$EvaluationWorkers = 1,
    [int]$EvaluationWorkerThreads = 2,
    [int]$CheckpointEvery = 10000,
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
    [double]$MaxQAbs = 100.0,
    [switch]$Smoke,
    [switch]$SaveVideo,
    [switch]$OnlineWandb
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

if ($Smoke) {
    $TotalUpdates = 2
    $EvalEvery = 1
    $ValidationEvery = 1
    $ValidationEpisodes = 1
    $TestEpisodes = 1
    $BatchSize = 32
    $ReplaySize = 64
    $MaxExpertSamples = 64
    $CheckpointEvery = 1
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $suffix = if ($Smoke) { "smoke" } else { "local" }
    $RunName = "iqlearn_recurrent_transformer_{0}_{1}" -f $suffix, (Get-Date -Format "yyyyMMdd_HHmmss")
}

$env:PYTHONPATH = "{0}{1}{2}" -f $RepoDir, [IO.Path]::PathSeparator, $env:PYTHONPATH
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "$WorkerThreads"
$env:MKL_NUM_THREADS = "$WorkerThreads"
$env:OPENBLAS_NUM_THREADS = "$WorkerThreads"
$env:NUMEXPR_NUM_THREADS = "$WorkerThreads"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

New-Item -ItemType Directory -Force -Path "logs", "logs/iq_learn" | Out-Null

Write-Host "Repository: $RepoDir"
Write-Host "Python: $Python"
Write-Host "Run name: $RunName"
Write-Host "Expert data: $ExpertData"
Write-Host "Policy: recurrent_transformer, hidden=$HiddenSize layers=$TransformerLayers heads=$TransformerHeads memory_tokens=$TransformerMemoryTokens context=$TransformerMemoryContextLength"
Write-Host "Memory profile: batch=$BatchSize replay_device=$ReplayDevice replay_size=$ReplaySize max_expert=$MaxExpertSamples"
Write-Host "Evaluation threading: workers=$EvaluationWorkers worker_threads=$EvaluationWorkerThreads"
Write-Host "Validation: every $ValidationEvery updates on split '$ValidationSplit', episodes=$ValidationEpisodes"
Write-Host "Test: final split '$TestSplit', episodes=$TestEpisodes"

& $Python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) {
    throw "Python/Torch CUDA check failed."
}

$wandbMode = if ($OnlineWandb) { "online" } else { "disabled" }
$videoArg = if ($SaveVideo) { "--save-checkpoint-video" } else { "--no-save-checkpoint-video" }

$TrainArgs = @(
    "scripts_gail/train_simple_iq_learn.py",
    "--expert-data", $ExpertData,
    "--run-name", $RunName,
    "--scene", $Scene,
    "--action-mode", "continuous",
    "--episode-root", $EpisodeRoot,
    "--prebuilt-split", $TrainSplit,
    "--validation-prebuilt-split", $ValidationSplit,
    "--test-prebuilt-split", $TestSplit,
    "--device", "cuda",
    "--replay-device", $ReplayDevice,
    "--pin-cpu-replay",
    "--policy-model", "recurrent_transformer",
    "--hidden-size", "$HiddenSize",
    "--transformer-layers", "$TransformerLayers",
    "--transformer-heads", "$TransformerHeads",
    "--transformer-dropout", "$TransformerDropout",
    "--transformer-temporal-kernel-size", "$TransformerTemporalKernelSize",
    "--transformer-temporal-layers", "$TransformerTemporalLayers",
    "--transformer-memory-tokens", "$TransformerMemoryTokens",
    "--transformer-memory-context-length", "$TransformerMemoryContextLength",
    "--transformer-recurrent-sequence-length", "$TransformerRecurrentSequenceLength",
    "--transformer-recurrent-sequences-per-batch", "$TransformerRecurrentSequencesPerBatch",
    "--transformer-recurrent-micro-batch-sequences", "$TransformerRecurrentMicroBatchSequences",
    "--transformer-use-causal-attention",
    "--torch-matmul-precision", "high",
    "--total-updates", "$TotalUpdates",
    "--eval-every", "$EvalEvery",
    "--eval-episodes", "$ValidationEpisodes",
    "--evaluation-num-workers", "$EvaluationWorkers",
    "--evaluation-worker-threads", "$EvaluationWorkerThreads",
    "--validation-every", "$ValidationEvery",
    "--validation-episodes", "$ValidationEpisodes",
    "--validation-control-all-vehicles",
    "--test-episodes", "$TestEpisodes",
    "--batch-size", "$BatchSize",
    "--replay-size", "$ReplaySize",
    "--max-expert-samples", "$MaxExpertSamples",
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
    "--checkpoint-every", "$CheckpointEvery",
    $videoArg,
    "--checkpoint-video-steps", "120",
    "--wandb-mode", $wandbMode,
    "--wandb-project", "highwayenv-ps-gail",
    "--wandb-tags", "iq-learn,continuous,recurrent-transformer,local-ngsim-env"
)

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "IQ-Learn temporal transformer training failed with exit code $LASTEXITCODE."
}

Write-Host "Done. Check logs/iq_learn/$RunName for best.pt, final.pt, and checkpoints."
