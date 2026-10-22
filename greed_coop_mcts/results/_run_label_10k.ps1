$env:PYTHONPATH='d:\compitition\MioVerse\final_version'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
Set-Location 'd:\compitition\MioVerse\final_version'
$log = 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\label_synth_10k.log'
$flag = 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\LABEL_10K_DONE.flag'
"==== RUN $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') pid=$PID ====" | Out-File -FilePath $log -Append -Encoding utf8
python -m greed_coop_mcts.results.label_synth_oracle --start 10001 --n 10000 --max-raise 4 --jobs 6 --wall-timeout 180 --batch-maps 100 --save-every 10 --out 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\facn_oracle_synth_10k.json' *>> $log
$code = $LASTEXITCODE
"LABEL_10K_EXIT exit=$code $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $log -Append -Encoding utf8
if ($code -eq 0) {
    'LABEL_10K_DONE' | Out-File -FilePath $log -Append -Encoding utf8
    'done' | Set-Content -Path $flag -Encoding ASCII
}
