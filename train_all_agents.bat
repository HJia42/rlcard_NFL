@echo off
REM Train PPO and DMC agents overnight
REM Run from the rlcard_NFL directory:
REM   .\train_all_agents.bat

echo ============================================
echo Starting Overnight Agent Training
echo Started at: %date% %time%
echo ============================================

cd /d %~dp0

REM 1. Train PPO with distribution model
echo.
echo [1/2] Training PPO Agent...
python examples/run_ppo_nfl.py --game nfl-bucketed --episodes 30000 ^
    --distribution-model ^
    --lr 0.001 --entropy-coef 0.1 ^
    --save-dir models/ppo_overnight

if %errorlevel% neq 0 (
    echo PPO training failed with error %errorlevel%
) else (
    echo PPO training completed successfully
)

REM 2. Train DMC with cached model
echo.
echo [2/2] Training DMC Agent...
python examples/run_dmc_nfl.py --game nfl-bucketed --cached-model ^
    --iterations 50000 ^
    --save-dir experiments/dmc_overnight

if %errorlevel% neq 0 (
    echo DMC training failed with error %errorlevel%
) else (
    echo DMC training completed successfully
)

echo.
echo ============================================
echo All Training Complete
echo Finished at: %date% %time%
echo ============================================
echo.
echo Models saved to:
echo   - models/ppo_overnight
echo   - experiments/dmc_overnight
echo.
pause
