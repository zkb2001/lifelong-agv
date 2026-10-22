$env:PYTHONPATH='d:\compitition\MioVerse\final_version'
Set-Location 'd:\compitition\MioVerse\final_version'
# Same renderer as MP4 (simulation/display.py): exactly 1 sim-second per frame at 3 fps.
python -m greed_coop_mcts.results._sh03_lifelong_display `
  --traj 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\trajectories\SH_custom_03_viz.csv' `
  --tasks 'd:\compitition\MioVerse\final_version\greed_coop_mcts\results\SH03_lifelong_seed400_task.csv' `
  --position 'd:\compitition\MioVerse\final_version\ml_research\map_editor\exports\SH_custom_03_position.csv' `
  --obstacles 'd:\compitition\MioVerse\final_version\ml_research\map_editor\exports\SH_custom_03.json' `
  --fps 3
