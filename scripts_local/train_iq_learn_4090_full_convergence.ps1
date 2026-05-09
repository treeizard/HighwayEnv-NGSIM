param(
    [string]$RunName = "",
    [switch]$Smoke,
    [switch]$OnlineWandb
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ScriptDir "train_iq_learn_4090.ps1"

if (-not (Test-Path $Launcher)) {
    throw "Missing launcher: $Launcher"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $suffix = if ($Smoke) { "smoke" } else { "full_convergence" }
    $RunName = "iqlearn_4090_{0}_{1}" -f $suffix, (Get-Date -Format "yyyyMMdd_HHmmss")
}

if ($Smoke) {
    & $Launcher `
        -RunName $RunName `
        -TotalUpdates 1 `
        -EvalEvery 1 `
        -EvalEpisodes 1 `
        -BatchSize 32 `
        -ReplaySize 64 `
        -MaxExpertSamples 64 `
        -CheckpointEvery 1 `
        -CheckpointVideoSteps 24 `
        -SaveVideo:($true) `
        -OnlineWandb:($OnlineWandb)
    exit $LASTEXITCODE
}

# Conservative long-run recipe for the local RTX 4090:
# - large CUDA replay/batches to keep the GPU busy;
# - behavior-cloning warmup to prevent early policy collapse;
# - clipped/regularized Q targets to catch IQ-Learn instability;
# - checkpoint videos every 5k updates so progress is inspectable.
& $Launcher `
    -RunName $RunName `
    -TotalUpdates 150000 `
    -EvalEvery 1000 `
    -EvalEpisodes 6 `
    -BatchSize 8192 `
    -ReplaySize 200000 `
    -MaxExpertSamples 200000 `
    -HiddenSize 256 `
    -WorkerThreads 4 `
    -CheckpointEvery 5000 `
    -CheckpointVideoSteps 200 `
    -LearningRate 7.5e-5 `
    -QLearningRate 7.5e-5 `
    -Gamma 0.95 `
    -IqAlpha 0.05 `
    -TargetTau 0.002 `
    -BcCoef 0.35 `
    -BcWarmupCoef 2.0 `
    -BcWarmupUpdates 10000 `
    -PolicyBcOnlyUpdates 2500 `
    -QL2Coef 0.002 `
    -QPolicyRegCoef 0.002 `
    -ConservativeQCoef 0.08 `
    -TargetEvalMeanLength 195 `
    -TargetBcLoss 0.02 `
    -MaxEvalCrashes 0 `
    -MaxQAbs 80 `
    -SaveVideo:($true) `
    -OnlineWandb:($OnlineWandb)

exit $LASTEXITCODE
