[CmdletBinding()]
param(
    [ValidateSet("qwen_cpp", "serveurperso", "audio_cpp", "official_python", "faster_python")]
    [string[]]$Implementations = @("qwen_cpp", "serveurperso", "audio_cpp", "official_python", "faster_python"),

    [ValidateSet("1.7b-base", "0.6b-base")]
    [string]$Variant = "1.7b-base",

    [ValidateSet("voice_clone", "basic")]
    [string]$Scenario = "voice_clone",

    [switch]$ValidateOnly,
    [int]$Runs = 3,
    [int]$MaxTokens = 128,
    [int]$Threads = 4,
    [string]$Text = "The quick brown fox jumps over the lazy dog. This is a benchmark for Qwen3 TTS implementations.",
    [string]$ReferenceAudio = "",
    [string]$ReferenceText = "",
    [string]$Language = "en",
    [int]$Seed = 42,
    [switch]$Greedy,
    [double]$Temperature = 0.9,
    [int]$TopK = 50,
    [double]$TopP = 1.0,
    [double]$RepetitionPenalty = 1.05,
    [string]$OutDir = "",
    [string]$WorkspaceRoot = "",

    [string]$QwenCppExe = "",
    [string]$QwenCppModels = "",
    [string]$QwenCppModelName = "",
    [string]$ServeurExe = "",
    [string]$ServeurTalker = "",
    [string]$ServeurCodec = "",
    [string]$AudioCppExe = "",
    [string]$AudioCppModel = "",
    [string]$AudioCppBackend = "cuda",
    [string]$AudioCppWeightType = "f16",
    [string]$PythonExe = "",
    [string]$OfficialRepo = "",
    [string]$FasterRepo = "",
    [string]$OfficialModel = "",
    [string]$FasterModel = "",
    [string]$PythonDeviceMap = "cuda",
    [ValidateSet("auto", "float32", "float16", "bfloat16")]
    [string]$PythonDType = "bfloat16"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "wav_stats.ps1")

function Resolve-OptionalPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if (Test-Path -LiteralPath $Path) {
        return (Resolve-Path -LiteralPath $Path).Path
    }
    return $Path
}

function Resolve-ExistingPath([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Find-FirstExisting([string[]]$Paths) {
    foreach ($path in $Paths) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return ""
}

function Get-HfSnapshot([string]$RepoCacheName) {
    $base = Join-Path $env:USERPROFILE ".cache\huggingface\hub\$RepoCacheName"
    $ref = Join-Path $base "refs\main"
    if (-not (Test-Path -LiteralPath $ref)) {
        return ""
    }
    $rev = (Get-Content -LiteralPath $ref -Raw).Trim()
    $snapshot = Join-Path $base "snapshots\$rev"
    if (Test-Path -LiteralPath $snapshot) {
        return (Resolve-Path -LiteralPath $snapshot).Path
    }
    return ""
}

function Convert-ToAudioCppLanguage([string]$Value) {
    $map = @{
        "en" = "english"; "zh" = "chinese"; "de" = "german"; "fr" = "french"
        "es" = "spanish"; "it" = "italian"; "ja" = "japanese"; "ko" = "korean"
        "pt" = "portuguese"; "ru" = "russian"
    }
    if ($map.ContainsKey($Value)) {
        return $map[$Value]
    }
    return $Value
}

function Convert-ToServeurLanguage([string]$Value) {
    $map = @{
        "en" = "English"; "zh" = "Chinese"; "de" = "German"; "fr" = "French"
        "es" = "Spanish"; "it" = "Italian"; "ja" = "Japanese"; "ko" = "Korean"
        "pt" = "Portuguese"; "ru" = "Russian"
    }
    if ($map.ContainsKey($Value)) {
        return $map[$Value]
    }
    return $Value
}

function Convert-ToPythonLanguage([string]$Value) {
    return Convert-ToServeurLanguage $Value
}

function Quote-CommandArgument([string]$Arg) {
    if ([string]::IsNullOrEmpty($Arg)) {
        return '""'
    }
    if ($Arg -notmatch '[\s"]') {
        return $Arg
    }
    return '"' + ($Arg -replace '"', '\"') + '"'
}

function Format-CommandLine([string]$Exe, [string[]]$CommandArgs) {
    return (Quote-CommandArgument $Exe) + " " + (($CommandArgs | ForEach-Object { Quote-CommandArgument $_ }) -join " ")
}

function Convert-ToInvariantDouble([string]$Value) {
    return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-RegexMetric([string]$Text, [string]$Pattern) {
    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($m.Success) {
        return Convert-ToInvariantDouble $m.Groups[1].Value
    }
    return $null
}

function Get-LastRegexMetric([string]$Text, [string]$Pattern) {
    $matches = [regex]::Matches($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($matches.Count -gt 0) {
        return Convert-ToInvariantDouble $matches[$matches.Count - 1].Groups[1].Value
    }
    return $null
}

function Get-BenchmarkInternalMetrics([string]$Implementation, [string]$LogText) {
    $metrics = [ordered]@{
        InternalTotalMs = $null
        InternalEncodeMs = $null
        InternalGenerateMs = $null
        InternalDecodeMs = $null
        InternalLoadMs = $null
        InternalTtfaMs = $null
    }

    if ($Implementation -eq "qwen_cpp") {
        $metrics.InternalLoadMs = Get-RegexMetric $LogText "All models loaded in\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalEncodeMs = Get-RegexMetric $LogText "^\s*Encode:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalGenerateMs = Get-RegexMetric $LogText "^\s*Generate:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalDecodeMs = Get-RegexMetric $LogText "^\s*Decode:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalTotalMs = Get-LastRegexMetric $LogText "^\s*Total:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
    } elseif ($Implementation -eq "serveurperso") {
        $metrics.InternalTotalMs = Get-RegexMetric $LogText "\[Perf\]\s+Total\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalEncodeMs = Get-RegexMetric $LogText "\[Perf\]\s+PromptBuild\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalGenerateMs = Get-RegexMetric $LogText "\[Perf\]\s+TalkerDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalDecodeMs = Get-RegexMetric $LogText "\[Perf\]\s+CodecDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
        $metrics.InternalTtfaMs = Get-RegexMetric $LogText "\[Perf\]\s+TTFA\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
    } elseif ($Implementation -eq "audio_cpp") {
        $metrics.InternalTotalMs = Get-RegexMetric $LogText "session\.wall_ms\s+([0-9]+(?:\.[0-9]+)?)"
        $metrics.InternalEncodeMs = Get-RegexMetric $LogText "qwen3_tts\.voice_prompt_ms\s+([0-9]+(?:\.[0-9]+)?)"
        $metrics.InternalGenerateMs = Get-RegexMetric $LogText "qwen3_tts\.talker_ms\s+([0-9]+(?:\.[0-9]+)?)"
        $metrics.InternalDecodeMs = Get-RegexMetric $LogText "qwen3_tts\.speech_decoder_ms\s+([0-9]+(?:\.[0-9]+)?)"
    } elseif ($Implementation -like "*_python") {
        $m = [regex]::Match($LogText, "BENCHMARK_JSON\s+(\{.+\})")
        if ($m.Success) {
            try {
                $json = $m.Groups[1].Value | ConvertFrom-Json
                $metrics.InternalLoadMs = [double]$json.load_seconds * 1000.0
                $metrics.InternalTotalMs = [double]$json.wall_seconds * 1000.0
                $metrics.InternalGenerateMs = [double]$json.synth_seconds * 1000.0
            } catch {
                # Leave metrics empty when the worker failed before JSON output.
            }
        }
    }

    return [PSCustomObject]$metrics
}

function Invoke-GitText([string]$Repo, [string[]]$GitArgs) {
    if (-not (Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
        return ""
    }
    $text = & git -C $Repo @GitArgs 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return ($text | Out-String).Trim()
}

function Get-GitSummary([string]$Repo) {
    if ([string]::IsNullOrWhiteSpace($Repo) -or -not (Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
        return [PSCustomObject]@{
            Repo = $Repo; IsGit = $false; Branch = ""; Commit = ""; Dirty = $null
            Upstream = ""; Ahead = $null; Behind = $null; Remote = ""
        }
    }
    $branch = Invoke-GitText $Repo @("branch", "--show-current")
    $commit = Invoke-GitText $Repo @("rev-parse", "--short", "HEAD")
    $remote = Invoke-GitText $Repo @("remote", "get-url", "origin")
    $upstream = Invoke-GitText $Repo @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    $ahead = $null
    $behind = $null
    if ($upstream) {
        $ab = Invoke-GitText $Repo @("rev-list", "--left-right", "--count", "HEAD...@{u}")
        if ($ab -match "^\s*(\d+)\s+(\d+)\s*$") {
            $ahead = [int]$Matches[1]
            $behind = [int]$Matches[2]
        }
    }
    $dirtyText = Invoke-GitText $Repo @("status", "--porcelain")
    $dirty = if ([string]::IsNullOrWhiteSpace($dirtyText)) { 0 } else { @($dirtyText -split "`r?`n" | Where-Object { $_ }).Count }
    return [PSCustomObject]@{
        Repo = $Repo; IsGit = $true; Branch = $branch; Commit = $commit; Dirty = $dirty
        Upstream = $upstream; Ahead = $ahead; Behind = $behind; Remote = $remote
    }
}

function New-PreflightRow([string]$Name, [string]$Repo, [string]$Executable, [string]$Model, [string]$Notes) {
    $git = Get-GitSummary $Repo
    $ok = $true
    $missing = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Executable) -and -not (Test-Path -LiteralPath $Executable)) {
        $ok = $false
        $missing.Add("exe") | Out-Null
    }
    if (-not [string]::IsNullOrWhiteSpace($Model) -and -not (Test-Path -LiteralPath $Model)) {
        $ok = $false
        $missing.Add("model") | Out-Null
    }
    [PSCustomObject]@{
        Implementation = $Name
        Ready = if ($ok) { "yes" } else { "no" }
        Branch = $git.Branch
        Commit = $git.Commit
        Dirty = $git.Dirty
        Ahead = $git.Ahead
        Behind = $git.Behind
        Executable = $Executable
        Model = $Model
        Missing = ($missing -join ",")
        Notes = $Notes
    }
}

function Invoke-BenchmarkCommand([string]$Name, [string]$Exe, [string[]]$CommandArgs, [string]$WorkingDirectory, [string]$LogPath, [string]$StdinText) {
    $commandLine = Format-CommandLine $Exe $CommandArgs
    $argumentLine = ($CommandArgs | ForEach-Object { Quote-CommandArgument $_ }) -join " "
    $stdoutPath = Join-Path $env:TEMP ("qwen3tts_bench_stdout_{0}.log" -f ([guid]::NewGuid().ToString("N")))
    $stderrPath = Join-Path $env:TEMP ("qwen3tts_bench_stderr_{0}.log" -f ([guid]::NewGuid().ToString("N")))
    $stdinPath = ""
    if (-not [string]::IsNullOrEmpty($StdinText)) {
        $stdinPath = Join-Path $env:TEMP ("qwen3tts_bench_stdin_{0}.txt" -f ([guid]::NewGuid().ToString("N")))
        Set-Content -LiteralPath $stdinPath -Value $StdinText -NoNewline -Encoding UTF8
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $startArgs = @{
            FilePath = $Exe
            ArgumentList = $argumentLine
            WorkingDirectory = $WorkingDirectory
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
            PassThru = $true
            Wait = $true
            WindowStyle = "Hidden"
        }
        if ($stdinPath) {
            $startArgs.RedirectStandardInput = $stdinPath
        }
        $proc = Start-Process @startArgs
        $exitCode = $proc.ExitCode
    } finally {
        $sw.Stop()
    }

    $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
    $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    $output = @()
    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        $output += "[stdout]"
        $output += $stdout.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        $output += "[stderr]"
        $output += $stderr.TrimEnd()
    }
    $outputText = $output -join [Environment]::NewLine

    if (Test-Path -LiteralPath $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force }
    if (Test-Path -LiteralPath $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force }
    if ($stdinPath -and (Test-Path -LiteralPath $stdinPath)) { Remove-Item -LiteralPath $stdinPath -Force }

    $logDir = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    @(
        "Implementation: $Name",
        "Command: $commandLine",
        "ExitCode: $exitCode",
        "WallSeconds: $($sw.Elapsed.TotalSeconds)",
        "",
        "[output]",
        $outputText
    ) | Set-Content -LiteralPath $LogPath -Encoding UTF8

    return [PSCustomObject]@{
        Name = $Name
        ExitCode = $exitCode
        WallSeconds = $sw.Elapsed.TotalSeconds
        LogText = $outputText
        CommandLine = $commandLine
    }
}

function New-ResultRow([string]$Implementation, [int]$Run, [string]$OutWav, [object]$CommandResult, [string]$RepoPath, [string]$ModelNote, [string]$LogPath) {
    $stats = if (Test-Path -LiteralPath $OutWav) { Get-WavAudioStats -Path $OutWav } else { New-EmptyWavAudioStats -path $OutWav -errorMessage "file not found" }
    $audioStatus = Get-WavAudioQualityStatus -Stats $stats
    $audioSeconds = if ($stats.Valid) { $stats.DurationSec } else { 0.0 }
    $rtf = if ($CommandResult.WallSeconds -gt 0) { $audioSeconds / $CommandResult.WallSeconds } else { 0.0 }
    $git = Get-GitSummary $RepoPath
    $internal = Get-BenchmarkInternalMetrics -Implementation $Implementation -LogText $CommandResult.LogText
    [PSCustomObject]@{
        Implementation = $Implementation
        Variant = $Variant
        Scenario = $Scenario
        Run = $Run
        ExitCode = $CommandResult.ExitCode
        AudioStatus = $audioStatus
        AudioSeconds = [math]::Round($audioSeconds, 3)
        WallSeconds = [math]::Round($CommandResult.WallSeconds, 3)
        RTF_AudioPerWall = [math]::Round($rtf, 3)
        InternalTotalMs = if ($null -ne $internal.InternalTotalMs) { [math]::Round($internal.InternalTotalMs, 1) } else { $null }
        InternalLoadMs = if ($null -ne $internal.InternalLoadMs) { [math]::Round($internal.InternalLoadMs, 1) } else { $null }
        InternalEncodeMs = if ($null -ne $internal.InternalEncodeMs) { [math]::Round($internal.InternalEncodeMs, 1) } else { $null }
        InternalGenerateMs = if ($null -ne $internal.InternalGenerateMs) { [math]::Round($internal.InternalGenerateMs, 1) } else { $null }
        InternalDecodeMs = if ($null -ne $internal.InternalDecodeMs) { [math]::Round($internal.InternalDecodeMs, 1) } else { $null }
        InternalTtfaMs = if ($null -ne $internal.InternalTtfaMs) { [math]::Round($internal.InternalTtfaMs, 1) } else { $null }
        Peak = if ($stats.Valid) { [math]::Round($stats.Peak, 8) } else { $null }
        Rms = if ($stats.Valid) { [math]::Round($stats.Rms, 8) } else { $null }
        NonZeroSamples = if ($stats.Valid) { $stats.NonZeroSamples } else { $null }
        RepoCommit = $git.Commit
        RepoDirty = $git.Dirty
        Model = $ModelNote
        Command = $CommandResult.CommandLine
        LogPath = $LogPath
        OutputWav = $OutWav
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = (Split-Path -Parent $repoRoot)
}
$WorkspaceRoot = (Resolve-Path -LiteralPath $WorkspaceRoot).Path

if ([string]::IsNullOrWhiteSpace($ReferenceAudio)) {
    $candidate = Join-Path $WorkspaceRoot "ref_audio_pcm.wav"
    if (-not (Test-Path -LiteralPath $candidate)) {
        $candidate = Join-Path $repoRoot "examples\readme_clone_input.wav"
    }
    $ReferenceAudio = $candidate
}
if ([string]::IsNullOrWhiteSpace($ReferenceText)) {
    $sidecar = [System.IO.Path]::ChangeExtension($ReferenceAudio, ".txt")
    if (Test-Path -LiteralPath $sidecar) {
        $ReferenceText = (Get-Content -LiteralPath $sidecar -Raw).Trim()
    } else {
        $referenceTextFile = Join-Path $repoRoot "reference_text.txt"
        if (Test-Path -LiteralPath $referenceTextFile) {
            $ReferenceText = (Get-Content -LiteralPath $referenceTextFile -Raw).Trim()
        }
    }
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = Join-Path $WorkspaceRoot "benchmark_output\framework_compare\$stamp"
}

if ([string]::IsNullOrWhiteSpace($QwenCppExe)) {
    $QwenCppExe = Find-FirstExisting @(
        (Join-Path $repoRoot "build-timing\qwen3-tts-cli.exe"),
        (Join-Path $repoRoot "build\qwen3-tts-cli.exe"),
        (Join-Path $repoRoot "build-cuda-ninja\qwen3-tts-cli.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($QwenCppModels)) {
    $QwenCppModels = Join-Path $repoRoot "models"
}
if ([string]::IsNullOrWhiteSpace($QwenCppModelName)) {
    $QwenCppModelName = if ($Variant -eq "1.7b-base") { "qwen3-tts-1.7b-base-f16.gguf" } else { "qwen3-tts-0.6b-f16.gguf" }
}

$serveurRepo = Join-Path $WorkspaceRoot "qwentts.cpp-serveurperso"
if ([string]::IsNullOrWhiteSpace($ServeurExe)) {
    $ServeurExe = Find-FirstExisting @(
        (Join-Path $serveurRepo "build-sm120\Release\qwen-tts.exe"),
        (Join-Path $serveurRepo "build\Release\qwen-tts.exe"),
        (Join-Path $serveurRepo "build\qwen-tts.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($ServeurTalker)) {
    $ServeurTalker = Join-Path $serveurRepo "models\qwen-talker-1.7b-base-Q8_0.gguf"
}
if ([string]::IsNullOrWhiteSpace($ServeurCodec)) {
    $ServeurCodec = Join-Path $serveurRepo "models\qwen-tokenizer-12hz-Q8_0.gguf"
}

$audioCppRepo = Join-Path $WorkspaceRoot "audio.cpp"
if ([string]::IsNullOrWhiteSpace($AudioCppExe)) {
    $AudioCppExe = Find-FirstExisting @(
        (Join-Path $audioCppRepo "build\windows-cuda-release\bin\audiocpp_cli.exe"),
        (Join-Path $audioCppRepo "build\windows-cpu-release\bin\audiocpp_cli.exe"),
        (Join-Path $audioCppRepo "build\bin\audiocpp_cli.exe")
    )
}
if ([string]::IsNullOrWhiteSpace($AudioCppModel)) {
    $cacheName = if ($Variant -eq "1.7b-base") { "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" } else { "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" }
    $AudioCppModel = Get-HfSnapshot $cacheName
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Find-FirstExisting @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path $WorkspaceRoot "Qwen3-TTS\.venv\Scripts\python.exe")
    )
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = "python"
    }
}
if ([string]::IsNullOrWhiteSpace($OfficialRepo)) {
    $OfficialRepo = Join-Path $WorkspaceRoot "Qwen3-TTS"
}
if ([string]::IsNullOrWhiteSpace($FasterRepo)) {
    $FasterRepo = Join-Path $WorkspaceRoot "faster-qwen3-tts"
}
if ([string]::IsNullOrWhiteSpace($OfficialModel)) {
    $OfficialModel = if ($Variant -eq "1.7b-base") { "Qwen/Qwen3-TTS-12Hz-1.7B-Base" } else { "Qwen/Qwen3-TTS-12Hz-0.6B-Base" }
}
if ([string]::IsNullOrWhiteSpace($FasterModel)) {
    $FasterModel = $OfficialModel
}

$qwenCppModelPath = Join-Path $QwenCppModels $QwenCppModelName
$preflight = @(
    New-PreflightRow "qwen_cpp" $repoRoot $QwenCppExe $qwenCppModelPath "GGUF via local CLI"
    New-PreflightRow "serveurperso" $serveurRepo $ServeurExe $ServeurTalker "Serveurperso GGUF talker; codec=$ServeurCodec"
    New-PreflightRow "audio_cpp" $audioCppRepo $AudioCppExe $AudioCppModel "HF model directory; weight_type=$AudioCppWeightType"
    New-PreflightRow "official_python" $OfficialRepo $PythonExe $OfficialRepo "imports qwen_tts from repo path"
    New-PreflightRow "faster_python" $FasterRepo $PythonExe $FasterRepo "imports faster_qwen3_tts from repo path"
)

Write-Host "Benchmark framework preflight" -ForegroundColor Cyan
Write-Host "  Workspace: $WorkspaceRoot"
Write-Host "  Variant:   $Variant"
Write-Host "  Scenario:  $Scenario"
Write-Host "  OutDir:    $OutDir"
if ($Scenario -eq "voice_clone") {
    Write-Host "  Ref audio: $ReferenceAudio"
    Write-Host "  Ref text:  $ReferenceText"
}
Write-Host ""
$preflight | Where-Object { $Implementations -contains $_.Implementation } |
    Format-Table Implementation, Ready, Branch, Commit, Dirty, Ahead, Behind, Missing, Notes -AutoSize

if ($ValidateOnly) {
    Write-Host ""
    Write-Host "ValidateOnly completed. No synthesis or benchmark process was started." -ForegroundColor Green
    return
}

if ($Scenario -eq "voice_clone") {
    $ReferenceAudio = Resolve-ExistingPath $ReferenceAudio "Reference audio"
    if ([string]::IsNullOrWhiteSpace($ReferenceText)) {
        throw "Voice-clone benchmarks require -ReferenceText or a sidecar/reference_text.txt file."
    }
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$logDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$serveurReferenceTextPath = ""
if ($Scenario -eq "voice_clone" -and -not [string]::IsNullOrWhiteSpace($ReferenceText)) {
    $serveurReferenceTextPath = Join-Path $OutDir "reference_text.txt"
    Set-Content -LiteralPath $serveurReferenceTextPath -Value $ReferenceText -Encoding UTF8
}

$rows = New-Object System.Collections.Generic.List[object]
$temperatureArg = if ($Greedy) { 0.0 } else { $Temperature }
$topKArg = if ($Greedy) { 1 } else { $TopK }
$topPArg = if ($Greedy) { 1.0 } else { $TopP }

for ($run = 1; $run -le $Runs; $run++) {
    if ($Implementations -contains "qwen_cpp") {
        $exe = Resolve-ExistingPath $QwenCppExe "qwen3-tts.cpp CLI"
        $models = Resolve-ExistingPath $QwenCppModels "qwen3-tts.cpp model directory"
        [void](Resolve-ExistingPath (Join-Path $models $QwenCppModelName) "qwen3-tts.cpp GGUF")
        $outWav = Join-Path $OutDir ("qwen_cpp_{0}_run{1}.wav" -f $Variant, $run)
        $logPath = Join-Path $logDir ("qwen_cpp_{0}_run{1}.log" -f $Variant, $run)
        $args = @("-m", $models, "--model-name", $QwenCppModelName, "-t", $Text, "-o", $outWav, "--max-tokens", "$MaxTokens", "--temperature", "$temperatureArg", "--top-k", "$topKArg", "--top-p", "$topPArg", "--repetition-penalty", "$RepetitionPenalty", "-l", $Language, "-j", "$Threads")
        if ($Scenario -eq "voice_clone") {
            $args += @("-r", $ReferenceAudio, "--reference-text", $ReferenceText)
        }
        Write-Host "[$run/$Runs] qwen_cpp" -ForegroundColor Yellow
        $cmd = Invoke-BenchmarkCommand "qwen_cpp" $exe $args $repoRoot $logPath ""
        $rows.Add((New-ResultRow "qwen_cpp" $run $outWav $cmd $repoRoot $QwenCppModelName $logPath)) | Out-Null
    }

    if ($Implementations -contains "serveurperso") {
        if ($Variant -ne "1.7b-base") {
            Write-Warning "serveurperso default checkout only has 1.7b-base Q8_0 models; skipping $Variant."
        } else {
            $exe = Resolve-ExistingPath $ServeurExe "Serveurperso qwen-tts CLI"
            $talker = Resolve-ExistingPath $ServeurTalker "Serveurperso talker GGUF"
            $codec = Resolve-ExistingPath $ServeurCodec "Serveurperso codec GGUF"
            $outWav = Join-Path $OutDir ("serveurperso_{0}_run{1}.wav" -f $Variant, $run)
            $logPath = Join-Path $logDir ("serveurperso_{0}_run{1}.log" -f $Variant, $run)
            $args = @("--model", $talker, "--codec", $codec, "-o", $outWav, "--lang", (Convert-ToServeurLanguage $Language), "--max-new", "$MaxTokens", "--seed", "$Seed")
            if ($Greedy) {
                $args += "--greedy"
            } else {
                $args += @("--temp", "$Temperature", "--top-k", "$TopK", "--top-p", "$TopP", "--rep-pen", "$RepetitionPenalty")
            }
            if ($Scenario -eq "voice_clone") {
                $args += @("--ref-wav", $ReferenceAudio)
                if ($serveurReferenceTextPath) {
                    $args += @("--ref-text", $serveurReferenceTextPath)
                }
            }
            Write-Host "[$run/$Runs] serveurperso" -ForegroundColor Yellow
            $cmd = Invoke-BenchmarkCommand "serveurperso" $exe $args $serveurRepo $logPath $Text
            $rows.Add((New-ResultRow "serveurperso" $run $outWav $cmd $serveurRepo (Split-Path -Leaf $talker) $logPath)) | Out-Null
        }
    }

    if ($Implementations -contains "audio_cpp") {
        $exe = Resolve-ExistingPath $AudioCppExe "audio.cpp CLI"
        $model = Resolve-ExistingPath $AudioCppModel "audio.cpp HF model directory"
        $outWav = Join-Path $OutDir ("audio_cpp_{0}_run{1}.wav" -f $Variant, $run)
        $logPath = Join-Path $logDir ("audio_cpp_{0}_run{1}.log" -f $Variant, $run)
        $args = @("--task", "tts", "--family", "qwen3_tts", "--model", $model, "--backend", $AudioCppBackend, "--mode", "offline", "--threads", "$Threads", "--text", $Text, "--out", $outWav, "--language", (Convert-ToAudioCppLanguage $Language), "--max-tokens", "$MaxTokens", "--seed", "$Seed", "--log")
        if ($Greedy) {
            $args += @("--do-sample", "false")
        } else {
            $args += @("--do-sample", "true", "--temperature", "$Temperature", "--top-k", "$TopK", "--top-p", "$TopP", "--repetition-penalty", "$RepetitionPenalty")
        }
        if ($Scenario -eq "voice_clone") {
            $args += @("--voice-ref", $ReferenceAudio, "--reference-text", $ReferenceText)
        }
        if (-not [string]::IsNullOrWhiteSpace($AudioCppWeightType)) {
            $args += @("--session-option", "qwen3_tts.weight_type=$AudioCppWeightType")
        }
        Write-Host "[$run/$Runs] audio_cpp" -ForegroundColor Yellow
        $cmd = Invoke-BenchmarkCommand "audio_cpp" $exe $args $audioCppRepo $logPath ""
        $rows.Add((New-ResultRow "audio_cpp" $run $outWav $cmd $audioCppRepo ("hf; weight_type=$AudioCppWeightType") $logPath)) | Out-Null
    }

    foreach ($pyImpl in @("official_python", "faster_python")) {
        if ($Implementations -notcontains $pyImpl) {
            continue
        }
        if ($Scenario -ne "voice_clone") {
            Write-Warning "$pyImpl currently supports only -Scenario voice_clone in this harness."
            continue
        }
        $repo = if ($pyImpl -eq "official_python") { Resolve-ExistingPath $OfficialRepo "official Qwen3-TTS repo" } else { Resolve-ExistingPath $FasterRepo "faster-qwen3-tts repo" }
        $backend = if ($pyImpl -eq "official_python") { "official" } else { "faster" }
        $model = if ($pyImpl -eq "official_python") { $OfficialModel } else { $FasterModel }
        $outWav = Join-Path $OutDir ("{0}_{1}_run{2}.wav" -f $pyImpl, $Variant, $run)
        $logPath = Join-Path $logDir ("{0}_{1}_run{2}.log" -f $pyImpl, $Variant, $run)
        $helper = Join-Path $PSScriptRoot "benchmark_python_framework.py"
        $args = @($helper, "--backend", $backend, "--repo", $repo, "--model", $model, "--output", $outWav, "--text", $Text, "--language", (Convert-ToPythonLanguage $Language), "--reference-audio", $ReferenceAudio, "--reference-text", $ReferenceText, "--max-tokens", "$MaxTokens", "--temperature", "$Temperature", "--top-k", "$TopK", "--top-p", "$TopP", "--repetition-penalty", "$RepetitionPenalty", "--seed", "$Seed", "--device-map", $PythonDeviceMap, "--dtype", $PythonDType, "--xvec-only", "--non-streaming-mode")
        if ($Greedy) {
            $args += "--greedy"
        }
        Write-Host "[$run/$Runs] $pyImpl" -ForegroundColor Yellow
        $cmd = Invoke-BenchmarkCommand $pyImpl $PythonExe $args $repo $logPath ""
        $rows.Add((New-ResultRow $pyImpl $run $outWav $cmd $repo $model $logPath)) | Out-Null
    }
}

$rawCsv = Join-Path $OutDir "framework_benchmark_raw.csv"
$rows | Export-Csv -NoTypeInformation -Path $rawCsv -Encoding UTF8

$summary = $rows |
    Group-Object Implementation |
    ForEach-Object {
        $pass = @($_.Group | Where-Object { $_.ExitCode -eq 0 -and $_.AudioStatus -eq "OK" })
        [PSCustomObject]@{
            Implementation = $_.Name
            Variant = $Variant
            Scenario = $Scenario
            Pass = ("{0}/{1}" -f $pass.Count, $_.Group.Count)
            WallSeconds = if ($pass.Count -gt 0) { [math]::Round(($pass | Measure-Object WallSeconds -Average).Average, 3) } else { 0.0 }
            AudioSeconds = if ($pass.Count -gt 0) { [math]::Round(($pass | Measure-Object AudioSeconds -Average).Average, 3) } else { 0.0 }
            RTF_AudioPerWall = if ($pass.Count -gt 0) { [math]::Round(($pass | Measure-Object RTF_AudioPerWall -Average).Average, 3) } else { 0.0 }
        }
    }
$summaryCsv = Join-Path $OutDir "framework_benchmark_summary.csv"
$summary | Export-Csv -NoTypeInformation -Path $summaryCsv -Encoding UTF8

Write-Host ""
Write-Host "Results" -ForegroundColor Green
$rows | Format-Table Implementation, Variant, Scenario, Run, ExitCode, AudioStatus, AudioSeconds, WallSeconds, RTF_AudioPerWall, Peak, Rms -AutoSize
Write-Host ""
Write-Host "Summary" -ForegroundColor Green
$summary | Format-Table Implementation, Variant, Scenario, Pass, WallSeconds, AudioSeconds, RTF_AudioPerWall -AutoSize
Write-Host ""
Write-Host "CSV:"
Write-Host "  $rawCsv"
Write-Host "  $summaryCsv"
