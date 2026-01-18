$env:PYTHONPATH = $PWD

# Standard Environment (nfl-bucketed)
Start-Job -Name "PPO_Standard" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo }
Start-Job -Name "NFSP_Standard" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp }
Start-Job -Name "DeepCFR_Standard" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr }

# Imperfect Information Environment (nfl-iig-bucketed)
Start-Job -Name "PPO_IIG" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo }
Start-Job -Name "NFSP_IIG" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp }
Start-Job -Name "DeepCFR_IIG" -ScriptBlock { python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr }

Write-Host "All training jobs started in background."
Write-Host "Use 'Get-Job' to see status and 'Receive-Job -Name <Name>' to see logs."
