# robust_financial_analysis.py
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

print("=== ROBUST FINANCIAL REGIME ANALYSIS ===")

class RobustFinancialAnalyzer:
    def __init__(self, symbol, start_date, end_date, n_regimes=2):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.n_regimes = n_regimes
        self.data = None
        self.model = None
        self.regimes = None
        
    def fetch_data(self):
        """Download historical stock data with robust error handling"""
        print(f"Step 1: Downloading data for {self.symbol}...")
        try:
            stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            
            if stock_data.empty:
                print(f"✗ No data returned for symbol {self.symbol}")
                return False
            
            self.data = stock_data
            print(f"✓ Successfully downloaded {len(self.data)} days of data")
            
            # Show available columns
            print(f"Available columns: {list(self.data.columns)}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading data: {e}")
            return False
    
    def get_price_column(self):
        """Determine which price column to use"""
        available_columns = self.data.columns.tolist()
        
        # Priority order for price columns
        price_priority = ['Adj Close', 'Close', 'Open', 'High', 'Low']
        
        for col in price_priority:
            if col in available_columns:
                print(f"✓ Using '{col}' as price column")
                return col
        
        # If none of the standard columns exist, use the first numeric column
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            print(f"✓ Using '{numeric_columns[0]}' as price column")
            return numeric_columns[0]
        else:
            print("✗ No suitable price column found")
            return None
    
    def preprocess_data(self):
        """Calculate returns and technical indicators with robust column handling"""
        print("Step 2: Preprocessing data...")
        if self.data is None:
            print("✗ No data available")
            return False
        
        # Get the appropriate price column
        price_column = self.get_price_column()
        if price_column is None:
            return False
        
        # Calculate returns using the available price column
        self.data['Returns'] = self.data[price_column].pct_change()
        
        # Calculate volatility (rolling standard deviation)
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # Calculate moving averages
        self.data['MA_20'] = self.data[price_column].rolling(window=20).mean()
        self.data['MA_50'] = self.data[price_column].rolling(window=50).mean()
        
        # Drop NaN values
        initial_count = len(self.data)
        self.data = self.data.dropna()
        final_count = len(self.data)
        
        print(f"✓ Preprocessing complete. Final dataset: {final_count} records")
        print(f"  Removed {initial_count - final_count} rows with NaN values")
        
        # Show basic statistics
        print(f"  Average return: {self.data['Returns'].mean():.6f}")
        print(f"  Return volatility: {self.data['Returns'].std():.6f}")
        
        return True
    
    def extract_features(self):
        """Extract features for regime detection"""
        print("Step 3: Extracting features...")
        
        # Use multiple features for better regime detection
        features_df = pd.DataFrame()
        features_df['returns'] = self.data['Returns']
        features_df['volatility'] = self.data['Volatility']
        features_df['ma_ratio'] = self.data['MA_20'] / self.data['MA_50'] - 1
        
        # Add momentum indicators
        price_column = self.get_price_column()
        features_df['momentum_5'] = self.data[price_column] / self.data[price_column].shift(5) - 1
        features_df['momentum_20'] = self.data[price_column] / self.data[price_column].shift(20) - 1
        
        # Drop any remaining NaN values
        features_df = features_df.dropna()
        self.data = self.data.loc[features_df.index]  # Align main data with features
        
        print(f"✓ Features extracted: {list(features_df.columns)}")
        return features_df.values
    
    def fit_gmm(self):
        """Fit Gaussian Mixture Model for regime detection"""
        print("Step 4: Fitting Gaussian Mixture Model...")
        
        features = self.extract_features()
        
        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=1000
        )
        
        self.regimes = self.model.fit_predict(features)
        self.data['Regime'] = self.regimes
        
        print("✓ Gaussian Mixture Model fitted successfully!")
        return True
    
    def analyze_regimes(self):
        """Analyze the detected market regimes"""
        print("Step 5: Analyzing market regimes...")
        
        print("\n" + "="*60)
        print("MARKET REGIME ANALYSIS")
        print("="*60)
        
        regime_stats = []
        
        for regime in range(self.n_regimes):
            regime_data = self.data[self.data['Regime'] == regime]
            
            if len(regime_data) == 0:
                continue
                
            stats = {
                'regime': regime,
                'count': len(regime_data),
                'percentage': (len(regime_data) / len(self.data)) * 100,
                'mean_return': regime_data['Returns'].mean(),
                'volatility': regime_data['Returns'].std(),
                'positive_returns': (regime_data['Returns'] > 0).mean(),
                'avg_volatility': regime_data['Volatility'].mean()
            }
            regime_stats.append(stats)
        
        if not regime_stats:
            print("No regimes detected!")
            return []
        
        # Sort regimes by volatility for consistent interpretation
        regime_stats.sort(key=lambda x: x['volatility'])
        
        for stats in regime_stats:
            regime = stats['regime']
            
            # Determine regime type
            if stats['mean_return'] > 0.001:
                trend = "STRONGLY BULLISH"
            elif stats['mean_return'] > 0:
                trend = "BULLISH"
            elif stats['mean_return'] > -0.001:
                trend = "BEARISH"
            else:
                trend = "STRONGLY BEARISH"
            
            if stats == regime_stats[0]:  # Lowest volatility
                vol_type = "LOW VOLATILITY"
            elif stats == regime_stats[-1]:  # Highest volatility
                vol_type = "HIGH VOLATILITY"
            else:
                vol_type = "MEDIUM VOLATILITY"
            
            print(f"\nRegime {regime}: {trend}, {vol_type}")
            print(f"  Mean Return: {stats['mean_return']:.6f}")
            print(f"  Volatility: {stats['volatility']:.6f}")
            print(f"  Positive Days: {stats['positive_returns']:.1%}")
            print(f"  Duration: {stats['count']} days ({stats['percentage']:.1f}%)")
        
        return regime_stats
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("Step 6: Creating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['green', 'red', 'blue', 'orange']
        price_column = self.get_price_column()
        
        # Plot 1: Price with regime coloring
        for regime in range(self.n_regimes):
            mask = self.data['Regime'] == regime
            if np.any(mask):
                ax1.plot(self.data.index[mask], self.data[price_column][mask],
                        color=colors[regime % len(colors)], linewidth=2,
                        label=f'Regime {regime}')
        
        ax1.set_title(f'{self.symbol} - Price with Market Regimes', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns by regime
        for regime in range(self.n_regimes):
            mask = self.data['Regime'] == regime
            if np.any(mask):
                ax2.scatter(self.data.index[mask], self.data['Returns'][mask],
                           color=colors[regime % len(colors)], s=10, alpha=0.6,
                           label=f'Regime {regime}')
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Daily Returns by Market Regime', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Returns', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime timeline
        ax3.plot(self.data.index, self.data['Regime'], color='purple', linewidth=1)
        ax3.set_title('Market Regime Timeline', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Regime', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_yticks(range(self.n_regimes))
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns distribution by regime
        regime_returns = []
        regime_labels = []
        for regime in range(self.n_regimes):
            returns = self.data[self.data['Regime'] == regime]['Returns']
            if len(returns) > 0:
                regime_returns.append(returns)
                regime_labels.append(f'Regime {regime}')
        
        if regime_returns:
            ax4.boxplot(regime_returns, labels=regime_labels)
            ax4.set_title('Returns Distribution by Regime', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Returns', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Visualizations saved and displayed")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPLETE FINANCIAL REGIME ANALYSIS FOR {self.symbol}")
        print(f"{'='*70}")
        
        steps = [
            ("Data Collection", self.fetch_data),
            ("Data Preprocessing", self.preprocess_data),
            ("Model Fitting", self.fit_gmm),
            ("Regime Analysis", self.analyze_regimes),
            ("Visualization", self.visualize_results),
        ]
        
        for step_name, step_function in steps:
            print(f"\n▶ {step_name}")
            print("-" * 40)
            success = step_function()
            if not success:
                print(f"✗ {step_name} failed!")
                return False
        
        print(f"\n{'='*70}")
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        return True

# Test with multiple symbols
def test_multiple_symbols():
    """Test the analyzer with multiple symbols"""
    symbols = ['SPY', 'AAPL', 'BTC-USD', 'GC=F']  # S&P 500, Apple, Bitcoin, Gold
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"TESTING SYMBOL: {symbol}")
        print(f"{'='*70}")
        
        analyzer = RobustFinancialAnalyzer(
            symbol=symbol,
            start_date='2020-01-01',
            end_date='2024-01-01',
            n_regimes=2
        )
        
        analyzer.run_complete_analysis()

# Main execution
if __name__ == "__main__":
    # Test individual symbol
    analyzer = RobustFinancialAnalyzer(
        symbol='SPY',  # Change this to any symbol you want to test
        start_date='2020-01-01',
        end_date='2024-01-01',
        n_regimes=2
    )
    
    analyzer.run_complete_analysis()