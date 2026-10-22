# Detached watchdog: keep 10k oracle labeling alive until done.
# Independent of Cursor shells. Safe to leave one instance only.

$ErrorActionPreference = 'Continue'
$Root = 'd:\compitition\MioVerse\final_version'
$Results = Join-Path $Root 'greed_coop_mcts\results'
$OutJson = Join-Path $Results 'facn_oracle_synth_10k.json'
$Log = Join-Path $Results 'label_synth_10k.log'
$WatchLog = Join-Path $Results 'label_watchdog.log'
$RunScript = Join-Path $Results '_run_label_10k.ps1'
$DoneMarker = Join-Path $Results 'LABEL_10K_DONE.flag'
$PollSec = 45

function Write-Watch([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $WatchLog -Value $line -Encoding UTF8
}

function Get-LabelPids {
    Get-WmiObject Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match 'label_synth_oracle') } |
        ForEach-Object { [int]$_.ProcessId }
}

function Test-LabelDone {
    if (Test-Path $DoneMarker) { return $true }
    if (-not (Test-Path $OutJson)) { return $false }
    try {
        $raw = Get-Content -Path $OutJson -Raw -ErrorAction Stop
        if ($raw -notmatch '"best"') { return $false }
        # Prefer log marker
        if ((Test-Path $Log) -and (Select-String -Path $Log -Pattern 'LABEL_10K_DONE|ALL DONE labeled=' -Quiet -ErrorAction SilentlyContinue)) {
            return $true
        }
        $py = @'
import json,sys
p=sys.argv[1]
d=json.load(open(p,encoding="utf-8"))
best=len(d.get("best") or {})
batch=d.get("batch") or {}
print(best)
print(batch.get("i"), batch.get("n"))
print(d.get("pending_left"))
'@
        $tmp = Join-Path $env:TEMP 'label_watch_check.py'
        Set-Content -Path $tmp -Value $py -Encoding UTF8
        $out = & python $tmp $OutJson 2>$null
        if (-not $out) { return $false }
        $best = [int]$out[0]
        if ($best -ge 10000) { return $true }
        return $false
    } catch {
        return $false
    }
}

function Start-LabelOnce {
    if (-not (Test-Path $RunScript)) {
        Write-Watch "ERROR missing $RunScript"
        return
    }
    $p = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $RunScript) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -PassThru
    Write-Watch "started wrapper pid=$($p.Id)"
}

Write-Watch "watchdog start pid=$PID poll=${PollSec}s"
while ($true) {
    if (Test-LabelDone) {
        'done' | Set-Content -Path $DoneMarker -Encoding ASCII
        Write-Watch "DONE detected; exiting watchdog"
        break
    }
    $pids = @(Get-LabelPids)
    if ($pids.Count -gt 1) {
        # Keep oldest (smallest StartTime), kill extras to avoid JSON corruption
        $procs = $pids | ForEach-Object {
            $id = $_
            Get-Process -Id $id -ErrorAction SilentlyContinue
        } | Where-Object { $_ } | Sort-Object StartTime
        $keep = $procs[0].Id
        foreach ($pr in $procs | Select-Object -Skip 1) {
            Write-Watch "killing duplicate label pid=$($pr.Id) keep=$keep"
            Stop-Process -Id $pr.Id -Force -ErrorAction SilentlyContinue
        }
        $pids = @($keep)
    }
    if ($pids.Count -eq 0) {
        Write-Watch "labeler dead; restarting resume"
        Start-LabelOnce
    } else {
        $bestInfo = ''
        try {
            $bestInfo = & python -c "import json;d=json.load(open(r'$OutJson',encoding='utf-8'));b=d.get('batch') or {};print(len(d.get('best') or {}), b.get('i'), b.get('n'), d.get('pending_left'))" 2>$null
        } catch {}
        Write-Watch "ok label_pids=$($pids -join ',') progress=$bestInfo"
    }
    Start-Sleep -Seconds $PollSec
}
Write-Watch "watchdog exit"
