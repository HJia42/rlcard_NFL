import sys
sys.path.insert(0, '.')
from rlcard.utils.eval_utils import plot_eval_history, load_eval_history

csv_path = 'models/test_cfr_std_2/eval_log.csv'
fig_path = 'models/test_cfr_std_2/debug_fig.png'

print("Loading history...")
try:
    history = load_eval_history(csv_path)
    print("History loaded:", history)
except Exception as e:
    print("Error loading history:", e)
    sys.exit(1)

print("Plotting...")
try:
    plot_eval_history(csv_path, fig_path, title="Debug Plot")
    print(f"Plot saved to {fig_path}")
except Exception as e:
    print("Error plotting:", e)
