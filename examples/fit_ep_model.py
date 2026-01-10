"""
Expected Points (EP) Regression Analysis

This script:
1. Loads NFL play data
2. Fits an EP regression model controlling for score differential
3. Produces diagnostic plots to check regression assumptions
4. Generates a fitted model for use in the game

Usage:
    python examples/fit_ep_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Try importing statsmodels and plotting
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    print("Warning: statsmodels not installed. Run: pip install statsmodels")
    HAS_STATSMODELS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed for plots")
    HAS_PLOTTING = False


def load_data():
    """Load NFL play-by-play data."""
    possible_paths = [
        Path(__file__).parent.parent / "rlcard" / "games" / "nfl" / ".." / ".." / ".." / ".." / "Code" / "data" / "cleaned_nfl_rl_data.csv",
        Path.home() / "Projects" / "NFL_Playcalling" / "Code" / "data" / "cleaned_nfl_rl_data.csv",
        Path("C:/Users/jiaha/Projects/NFL_Playcalling/Code/data/cleaned_nfl_rl_data.csv"),
    ]
    
    for p in possible_paths:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Loaded {len(df):,} plays from {p}")
            return df
    
    raise FileNotFoundError("Could not find cleaned_nfl_rl_data.csv")


def prepare_features(df):
    """Prepare features for EP regression."""
    # Check what columns we have
    print("\n=== Available Columns ===")
    print([c for c in df.columns if any(x in c.lower() for x in ['ep', 'score', 'down', 'yard', 'qtr'])])
    
    # Required columns
    required = ['down', 'ydstogo', 'yardline_100']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create a working copy
    data = df.copy()
    
    # Filter to valid plays
    data = data[data['down'].between(1, 4)]
    data = data[data['ydstogo'].between(1, 50)]
    data = data[data['yardline_100'].between(1, 99)]
    
    # Check for EP column (from nflfastR data)
    ep_col = None
    for candidate in ['ep', 'epa', 'expected_points', 'ep_before']:
        if candidate in data.columns:
            ep_col = candidate
            break
    
    if ep_col is None:
        print("Warning: No EP column found. Creating proxy from scoring results...")
        # Create a proxy EP based on drive outcomes
        # This is less accurate but works if no EP data
        data['ep_target'] = data['yardline_100'].apply(
            lambda x: (100 - x) / 100 * 7  # Simple linear approximation
        )
        ep_col = 'ep_target'
    
    # Score differential (if available)
    if 'score_differential' in data.columns:
        data['score_diff'] = data['score_differential']
    elif 'posteam_score' in data.columns and 'defteam_score' in data.columns:
        data['score_diff'] = data['posteam_score'] - data['defteam_score']
    else:
        print("Warning: No score differential found, creating dummy column")
        data['score_diff'] = 0
    
    # Convert yardline to field position (yards from own goal)
    # yardline_100 is typically yards to opponent's goal, so:
    data['yardline'] = 100 - data['yardline_100']
    
    # Create dummy variables for down
    data['is_2nd_down'] = (data['down'] == 2).astype(int)
    data['is_3rd_down'] = (data['down'] == 3).astype(int)
    data['is_4th_down'] = (data['down'] == 4).astype(int)
    
    # Interaction terms
    data['yardline_sq'] = data['yardline'] ** 2
    data['ydstogo_sq'] = np.minimum(data['ydstogo'], 20) ** 2
    
    # Red zone indicator
    data['in_redzone'] = (data['yardline'] >= 80).astype(int)
    
    # Goal-to-go indicator (critical for capturing stripe pattern)
    if 'goal_to_go' in data.columns:
        data['is_goal_to_go'] = data['goal_to_go'].astype(int)
    else:
        data['is_goal_to_go'] = 0
    
    # Filter out rows with missing EP
    data = data.dropna(subset=[ep_col])
    
    print(f"\nFiltered to {len(data):,} valid plays")
    
    return data, ep_col


def fit_ep_model(data, ep_col):
    """Fit EP regression model."""
    if not HAS_STATSMODELS:
        print("Cannot fit model without statsmodels")
        return None
    
    # Define features
    feature_cols = [
        'yardline', 'yardline_sq',
        'ydstogo',
        'is_2nd_down', 'is_3rd_down', 'is_4th_down',
        'in_redzone',
        'is_goal_to_go',
        'score_diff'
    ]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    X = data[feature_cols].copy()
    y = data[ep_col].copy()
    
    # Add constant
    X = sm.add_constant(X)
    
    print("\n=== Fitting OLS Regression ===")
    print(f"Features: {feature_cols}")
    print(f"Target: {ep_col}")
    print(f"N observations: {len(y):,}")
    
    # Fit model
    model = sm.OLS(y, X).fit()
    
    # Print summary
    print("\n" + "=" * 60)
    print(model.summary())
    print("=" * 60)
    
    # VIF for multicollinearity check
    print("\n=== Variance Inflation Factors (VIF) ===")
    print("(VIF > 5 suggests multicollinearity)")
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data.to_string(index=False))
    
    return model, X, y


def plot_diagnostics(model, X, y, save_dir=None):
    """Create regression diagnostic plots."""
    if not HAS_PLOTTING:
        print("Cannot create plots without matplotlib/seaborn")
        return
    
    if save_dir is None:
        save_dir = Path(__file__).parent
    else:
        save_dir = Path(save_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fitted values and residuals
    fitted = model.fittedvalues
    residuals = model.resid
    
    # 1. Residuals vs Fitted (check linearity and homoscedasticity)
    ax1 = axes[0, 0]
    ax1.scatter(fitted, residuals, alpha=0.2, s=5)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted\n(Check: should be random around 0)')
    
    # Add LOWESS smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.1)
        ax1.plot(smoothed[:, 0], smoothed[:, 1], color='orange', linewidth=2)
    except:
        pass
    
    # 2. Q-Q Plot (check normality of residuals)
    ax2 = axes[0, 1]
    sm.qqplot(residuals, line='45', ax=ax2, alpha=0.3)
    ax2.set_title('Q-Q Plot\n(Check: points should follow diagonal)')
    
    # 3. Scale-Location (check homoscedasticity)
    ax3 = axes[0, 2]
    std_residuals = np.sqrt(np.abs(residuals / residuals.std()))
    ax3.scatter(fitted, std_residuals, alpha=0.2, s=5)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location\n(Check: should be flat)')
    
    # 4. Histogram of residuals
    ax4 = axes[1, 0]
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Residual Distribution\n(Mean={residuals.mean():.3f}, Std={residuals.std():.3f})')
    
    # 5. Coefficient plot
    ax5 = axes[1, 1]
    coef_df = pd.DataFrame({
        'Feature': model.params.index,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values
    })
    coef_df = coef_df[coef_df['Feature'] != 'const']  # Exclude intercept
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    ax5.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax5.axvline(x=0, color='black', linestyle='-')
    ax5.set_xlabel('Coefficient')
    ax5.set_title('Coefficient Magnitudes\n(Green=+, Red=-)')
    
    # 6. EP by Field Position (actual vs predicted)
    ax6 = axes[1, 2]
    # Sample for faster plotting
    sample_idx = np.random.choice(len(fitted), min(5000, len(fitted)), replace=False)
    yardline_sample = X.iloc[sample_idx]['yardline'].values if 'yardline' in X.columns else np.zeros(len(sample_idx))
    ax6.scatter(yardline_sample, y.iloc[sample_idx], alpha=0.2, s=5, label='Actual', color='blue')
    ax6.scatter(yardline_sample, fitted.iloc[sample_idx], alpha=0.2, s=5, label='Predicted', color='orange')
    ax6.set_xlabel('Yardline (from own goal)')
    ax6.set_ylabel('Expected Points')
    ax6.set_title('EP by Field Position')
    ax6.legend()
    
    plt.tight_layout()
    
    plot_path = save_dir / 'ep_regression_diagnostics.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved diagnostic plots to {plot_path}")
    plt.close()
    
    # Additional: Score differential effect plot
    if 'score_diff' in X.columns:
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        # Bin by score differential
        score_bins = pd.cut(X['score_diff'], bins=np.arange(-30, 35, 5))
        ep_by_score = pd.DataFrame({
            'score_bin': score_bins,
            'ep_actual': y.values,
            'ep_predicted': fitted.values
        }).groupby('score_bin').mean()
        
        x_pos = range(len(ep_by_score))
        ax.bar(x_pos, ep_by_score['ep_actual'], alpha=0.5, label='Actual EP')
        ax.plot(x_pos, ep_by_score['ep_predicted'], 'ro-', label='Predicted EP')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(x) for x in ep_by_score.index], rotation=45)
        ax.set_xlabel('Score Differential Bin')
        ax.set_ylabel('Expected Points')
        ax.set_title('EP by Score Differential\n(Shows score_diff coefficient effect)')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        score_plot_path = save_dir / 'ep_score_differential_effect.png'
        plt.savefig(score_plot_path, dpi=150)
        print(f"Saved score differential plot to {score_plot_path}")
        plt.close()


def create_ep_function(model, feature_cols):
    """Create a function to calculate EP with score_diff=0."""
    
    coefs = model.params.to_dict()
    
    def calculate_ep(down, ydstogo, yardline, score_diff=0):
        """
        Calculate expected points for a game state.
        
        Args:
            down: 1-4
            ydstogo: yards to first down
            yardline: yards from own goal (1-99)
            score_diff: score differential (set to 0 for neutral estimation)
        
        Returns:
            Expected points (-7 to 7 range)
        """
        # Build feature vector
        features = {
            'const': 1,
            'yardline': yardline,
            'yardline_sq': yardline ** 2,
            'ydstogo': min(ydstogo, 20),
            'is_2nd_down': 1 if down == 2 else 0,
            'is_3rd_down': 1 if down == 3 else 0,
            'is_4th_down': 1 if down == 4 else 0,
            'in_redzone': 1 if yardline >= 80 else 0,
            'score_diff': score_diff,
        }
        
        # Calculate EP
        ep = sum(coefs.get(k, 0) * v for k, v in features.items())
        
        # Clip to reasonable range
        return max(-7, min(7, ep))
    
    return calculate_ep


def save_model(model, feature_cols, save_path=None):
    """Save the fitted model for use in the game."""
    if save_path is None:
        save_path = Path(__file__).parent.parent / "rlcard" / "games" / "nfl" / "ep_model.pkl"
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'coefficients': model.params.to_dict(),
        'feature_cols': feature_cols,
        'r_squared': model.rsquared,
        'n_obs': int(model.nobs),
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nSaved EP model to {save_path}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Use score_diff=0 for neutral expected points estimation")


def main():
    print("=" * 60)
    print("Expected Points (EP) Regression Analysis")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    data, ep_col = prepare_features(df)
    
    # Fit model
    if HAS_STATSMODELS:
        model, X, y = fit_ep_model(data, ep_col)
        
        if model is not None:
            # Create diagnostic plots
            plot_diagnostics(model, X, y)
            
            # Show key findings
            print("\n=== Key Findings ===")
            coefs = model.params
            if 'score_diff' in coefs:
                print(f"Score Differential Effect: {coefs['score_diff']:.4f} EP per point")
                print("  -> Set score_diff=0 for neutral EP estimation")
            
            if 'yardline' in coefs:
                print(f"Yardline Effect: {coefs['yardline']:.4f} EP per yard")
            
            if 'is_4th_down' in coefs:
                print(f"4th Down Penalty: {coefs['is_4th_down']:.2f} EP")
            
            # Save model
            save_model(model, list(X.columns))
            
            # Create EP function
            calc_ep = create_ep_function(model, list(X.columns))
            
            # Demo
            print("\n=== EP Predictions (score_diff=0) ===")
            test_cases = [
                (1, 10, 25, "1st & 10 at own 25"),
                (1, 10, 50, "1st & 10 at midfield"),
                (1, 10, 80, "1st & 10 at opp 20"),
                (3, 10, 50, "3rd & 10 at midfield"),
                (4, 5, 65, "4th & 5 at opp 35"),
            ]
            for down, ydstogo, yardline, desc in test_cases:
                ep = calc_ep(down, ydstogo, yardline, score_diff=0)
                print(f"  {desc}: EP = {ep:.2f}")
    else:
        print("\nInstall statsmodels to run regression analysis:")
        print("  pip install statsmodels")


if __name__ == '__main__':
    main()
