import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - FIXES TclError
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ImprovedAnomalyVisualizer:
    def __init__(self, feature_file='../data/features.csv', 
                 sliding_file='../output/sliding_anomalies.csv',
                 heap_file='../output/heap_anomalies.csv',
                 plots_dir='plots'):
        self.feature_file = feature_file
        self.sliding_file = sliding_file
        self.heap_file = heap_file
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load feature data
        self.features_df = pd.read_csv(self.feature_file)
        print(f"📊 Feature data loaded: {len(self.features_df)} rows")
        print(f"📊 Columns in features data: {list(self.features_df.columns)}")
        
        # Load anomaly data
        try:
            self.sliding_df = pd.read_csv(self.sliding_file)
            self.heap_df = pd.read_csv(self.heap_file)
            print(f"🔍 Sliding anomalies: {len(self.sliding_df)}")
            print(f"🔍 Heap anomalies: {len(self.heap_df)}")
        except Exception as e:
            print(f"Error loading anomaly files: {e}")
            return
        
        # Clean and prepare data
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare data for visualization"""
        # Check if ticker column exists, if not create a dummy one or skip ticker-specific analysis
        if 'ticker' not in self.features_df.columns:
            print("Warning: 'ticker' column not found in features data")
            # Create a dummy ticker column based on row groups or use a generic ticker
            if 'Ticker' in self.features_df.columns:
                self.features_df['ticker'] = self.features_df['Ticker']
            else:
                # Create dummy tickers - assume every ~100 rows is a different stock
                n_tickers = max(1, len(self.features_df) // 100)
                ticker_names = [f"STOCK_{i:03d}" for i in range(n_tickers)]
                self.features_df['ticker'] = np.repeat(ticker_names, len(self.features_df) // n_tickers + 1)[:len(self.features_df)]
            print(f"Created ticker column with {self.features_df['ticker'].nunique()} unique tickers")
        
        # Convert dates if they exist
        date_columns = [col for col in self.features_df.columns if 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            self.features_df['date'] = pd.to_datetime(self.features_df[date_col])
            print(f"Using '{date_col}' as date column")
        
        # Merge anomaly data with features
        try:
            self.sliding_merged = self.features_df.iloc[self.sliding_df['index'].values].copy()
            self.heap_merged = self.features_df.iloc[self.heap_df['index'].values].copy()
            
            # Add anomaly flags
            self.features_df['is_sliding_anomaly'] = False
            self.features_df['is_heap_anomaly'] = False
            
            self.features_df.loc[self.sliding_df['index'], 'is_sliding_anomaly'] = True
            self.features_df.loc[self.heap_df['index'], 'is_heap_anomaly'] = True
            
            print("✅ Data preparation complete")
        except Exception as e:
            print(f"Error in data preparation: {e}")
            print("This might be due to index mismatch between anomaly files and features")
    
    def create_overview_plots(self):
        """Create overview comparison plots"""
        print("Creating overview plots...")
        
        try:
            # 1. Anomaly counts comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Stock Market Anomaly Detection - Overview', fontsize=16, fontweight='bold')
            
            # Anomaly counts bar chart
            methods = ['Sliding Window', 'Heap-Based']
            counts = [len(self.sliding_df), len(self.heap_df)]
            percentages = [c/len(self.features_df)*100 for c in counts]
            
            bars = ax1.bar(methods, counts, color=['#3498db', '#e74c3c'], alpha=0.8)
            ax1.set_title('Total Anomalies Detected', fontweight='bold')
            ax1.set_ylabel('Number of Anomalies')
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                        f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # Detection rate comparison
            ax2.bar(methods, percentages, color=['#3498db', '#e74c3c'], alpha=0.8)
            ax2.set_title('Detection Rates (%)', fontweight='bold')
            ax2.set_ylabel('Percentage of Data')
            ax2.set_ylim(0, max(percentages) * 1.2)
            
            # Top tickers for sliding window
            if hasattr(self, 'sliding_merged') and len(self.sliding_merged) > 0:
                sliding_ticker_counts = self.sliding_merged['ticker'].value_counts().head(10)
                ax3.barh(range(len(sliding_ticker_counts)), sliding_ticker_counts.values, 
                        color='#3498db', alpha=0.8)
                ax3.set_yticks(range(len(sliding_ticker_counts)))
                ax3.set_yticklabels(sliding_ticker_counts.index)
                ax3.set_title('Top 10 Tickers - Sliding Window', fontweight='bold')
                ax3.set_xlabel('Number of Anomalies')
            else:
                ax3.text(0.5, 0.5, 'No sliding window data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Top 10 Tickers - Sliding Window', fontweight='bold')
            
            # Top tickers for heap
            if hasattr(self, 'heap_merged') and len(self.heap_merged) > 0:
                heap_ticker_counts = self.heap_merged['ticker'].value_counts().head(10)
                ax4.barh(range(len(heap_ticker_counts)), heap_ticker_counts.values, 
                        color='#e74c3c', alpha=0.8)
                ax4.set_yticks(range(len(heap_ticker_counts)))
                ax4.set_yticklabels(heap_ticker_counts.index)
                ax4.set_title('Top 10 Tickers - Heap-Based', fontweight='bold')
                ax4.set_xlabel('Number of Anomalies')
            else:
                ax4.text(0.5, 0.5, 'No heap data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Top 10 Tickers - Heap-Based', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'improved_anomaly_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Distribution comparison
            self.create_distribution_plots()
            
            print("✅ Overview plots created")
            
        except Exception as e:
            print(f"Error creating overview plots: {e}")
    
    def create_distribution_plots(self):
        """Create distribution comparison plots"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anomaly Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Get the actual feature values - try different column names
            feature_col = None
            possible_columns = ['daily_return', 'Daily Return', 'price_change', 'return', 'log_return', 'pct_change']
            
            for col in possible_columns:
                if col in self.features_df.columns:
                    feature_col = col
                    break
            
            if feature_col is None:
                # Try to find a numeric column that looks like returns
                numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
                feature_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if feature_col and hasattr(self, 'sliding_merged') and hasattr(self, 'heap_merged'):
                print(f"Using feature column: {feature_col}")
                
                # Distribution of normal vs anomaly values
                normal_values = self.features_df[~self.features_df['is_sliding_anomaly']][feature_col]
                sliding_anomaly_values = self.features_df[self.features_df['is_sliding_anomaly']][feature_col]
                heap_anomaly_values = self.features_df[self.features_df['is_heap_anomaly']][feature_col]
                
                # Histogram comparison
                ax1.hist(normal_values, bins=50, alpha=0.7, label='Normal', color='gray', density=True)
                if len(sliding_anomaly_values) > 0:
                    ax1.hist(sliding_anomaly_values, bins=30, alpha=0.8, label='Sliding Anomalies', 
                            color='#3498db', density=True)
                ax1.set_title('Value Distribution - Sliding Window', fontweight='bold')
                ax1.set_xlabel(feature_col.replace('_', ' ').title())
                ax1.set_ylabel('Density')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.hist(normal_values, bins=50, alpha=0.7, label='Normal', color='gray', density=True)
                if len(heap_anomaly_values) > 0:
                    ax2.hist(heap_anomaly_values, bins=30, alpha=0.8, label='Heap Anomalies', 
                            color='#e74c3c', density=True)
                ax2.set_title('Value Distribution - Heap-Based', fontweight='bold')
                ax2.set_xlabel(feature_col.replace('_', ' ').title())
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Box plots
                data_for_box = [normal_values.dropna().values]
                labels = ['Normal']
                colors = ['lightgray']
                
                if len(sliding_anomaly_values) > 0:
                    data_for_box.append(sliding_anomaly_values.dropna().values)
                    labels.append('Sliding\nAnomalies')
                    colors.append('#3498db')
                
                if len(heap_anomaly_values) > 0:
                    data_for_box.append(heap_anomaly_values.dropna().values)
                    labels.append('Heap\nAnomalies')
                    colors.append('#e74c3c')
                
                box_plot = ax3.boxplot(data_for_box, labels=labels, patch_artist=True)
                
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                
                ax3.set_title('Value Distribution Comparison', fontweight='bold')
                ax3.set_ylabel(feature_col.replace('_', ' ').title())
                ax3.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, f'Feature column not found.\nAvailable columns: {list(self.features_df.columns)[:5]}...', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'No feature data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax3.text(0.5, 0.5, 'No feature data available', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # Overlap analysis
            if hasattr(self, 'features_df') and 'is_sliding_anomaly' in self.features_df.columns:
                overlap_data = pd.crosstab(self.features_df['is_sliding_anomaly'], 
                                          self.features_df['is_heap_anomaly'], 
                                          margins=True)
                
                # Create a heatmap for overlap
                sns.heatmap(overlap_data.iloc[:-1, :-1], annot=True, fmt='d', 
                           cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'})
                ax4.set_title('Method Overlap Analysis', fontweight='bold')
                ax4.set_xlabel('Heap-Based Anomaly')
                ax4.set_ylabel('Sliding Window Anomaly')
            else:
                ax4.text(0.5, 0.5, 'No overlap data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Method Overlap Analysis', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'anomaly_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating distribution plots: {e}")
    
    def create_ticker_specific_plots(self, top_n=5):
        """Create detailed plots for specific tickers"""
        print(f"Creating plots for top {top_n} tickers...")

        try:
            if hasattr(self, 'sliding_merged') and hasattr(self, 'heap_merged'):
                sliding_tickers = self.sliding_merged['ticker'].value_counts()
                heap_tickers = self.heap_merged['ticker'].value_counts()

                # Combine both ticker counts and keep top N unique tickers
                combined_tickers = pd.concat([sliding_tickers, heap_tickers]) \
                                      .groupby(level=0).sum() \
                                      .sort_values(ascending=False)

                top_tickers = combined_tickers.head(top_n).index
                print(f"✅ Selected top tickers from both methods: {list(top_tickers)}")

                for ticker in top_tickers:
                    self.create_single_ticker_plot(ticker)

                print(f"✅ Created plots for {len(top_tickers)} tickers")
            else:
                print("⚠️ Not enough anomaly data to create ticker-specific plots")

        except Exception as e:
            print(f"Error creating ticker-specific plots: {e}")

    def create_single_ticker_plot(self, ticker):
        """Create a detailed plot for a single ticker"""
        try:
            # Get ticker data
            ticker_data = self.features_df[self.features_df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                return
            
            # Sort by date if available
            if 'date' in ticker_data.columns:
                ticker_data = ticker_data.sort_values('date')
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'{ticker} - Anomaly Detection Analysis', fontsize=16, fontweight='bold')
            
            # Get feature column
            feature_col = None
            possible_columns = ['daily_return', 'Daily Return', 'price_change', 'return', 'log_return', 'pct_change', 'close', 'price']
            
            for col in possible_columns:
                if col in ticker_data.columns:
                    feature_col = col
                    break
            
            if feature_col is None:
                numeric_cols = ticker_data.select_dtypes(include=[np.number]).columns
                feature_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if feature_col:
                x_axis = ticker_data.index if 'date' not in ticker_data.columns else ticker_data['date']
                
                # Masks
                normal_mask = ~(ticker_data['is_sliding_anomaly'] | ticker_data['is_heap_anomaly'])
                sliding_mask = ticker_data['is_sliding_anomaly']
                heap_only_mask = ticker_data['is_heap_anomaly'] & ~ticker_data['is_sliding_anomaly']
                
                # Time series scatter plot
                ax1.scatter(x_axis[normal_mask], ticker_data[normal_mask][feature_col], 
                            c='gray', alpha=0.6, s=20, label='Normal')
                if sliding_mask.any():
                    ax1.scatter(x_axis[sliding_mask], ticker_data[sliding_mask][feature_col], 
                                c='#3498db', s=60, label='Sliding Window', marker='^', edgecolors='darkblue')
                if heap_only_mask.any():
                    ax1.scatter(x_axis[heap_only_mask], ticker_data[heap_only_mask][feature_col], 
                                c='#e74c3c', s=40, label='Heap Only', marker='o', edgecolors='darkred')
                
                ax1.set_title(f'{ticker} - {feature_col.replace("_", " ").title()} Over Time')
                ax1.set_ylabel(feature_col.replace('_', ' ').title())
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                
                if sliding_mask.any() or heap_only_mask.any():
                    categories = ['Mean', 'Std', 'Min', 'Max']
                    x = np.arange(len(categories))
                    width = 0.25
                    offset = 0

                    all_data_stats = [ticker_data[feature_col].mean(), ticker_data[feature_col].std(),
                                      ticker_data[feature_col].min(), ticker_data[feature_col].max()]
                    ax2.bar(x + offset, all_data_stats, width, label='All Data', alpha=0.8, color='gray')
                    offset += width

                    if sliding_mask.any():
                        sliding_stats = [ticker_data[sliding_mask][feature_col].mean(), 
                                         ticker_data[sliding_mask][feature_col].std(),
                                         ticker_data[sliding_mask][feature_col].min(), 
                                         ticker_data[sliding_mask][feature_col].max()]
                        ax2.bar(x + offset, sliding_stats, width, label='Sliding Anomalies', alpha=0.8, color='#3498db')
                        offset += width

                    if heap_only_mask.any():
                        heap_stats = [ticker_data[heap_only_mask][feature_col].mean(), 
                                      ticker_data[heap_only_mask][feature_col].std(),
                                      ticker_data[heap_only_mask][feature_col].min(), 
                                      ticker_data[heap_only_mask][feature_col].max()]
                        ax2.bar(x + offset, heap_stats, width, label='Heap Anomalies', alpha=0.8, color='#e74c3c')

                    ax2.set_xlabel('Statistics')
                    ax2.set_ylabel('Value')
                    ax2.set_title(f'{ticker} - Statistical Comparison')
                    ax2.set_xticks(x + width)
                    ax2.set_xticklabels(categories)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No anomalies found for this ticker',
                             ha='center', va='center', transform=ax2.transAxes)

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{ticker}_detailed_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating plot for ticker {ticker}: {e}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("Creating summary report...")
        
        try:
            # Calculate overlap
            if hasattr(self, 'features_df') and 'is_sliding_anomaly' in self.features_df.columns:
                overlap_count = len(self.features_df[
                    self.features_df['is_sliding_anomaly'] & self.features_df['is_heap_anomaly']
                ])
                
                # Calculate detection rates
                sliding_rate = len(self.sliding_df) / len(self.features_df) * 100
                heap_rate = len(self.heap_df) / len(self.features_df) * 100
                overlap_rate = overlap_count / len(self.features_df) * 100
                
                # Top tickers analysis
                sliding_top_tickers = self.sliding_merged['ticker'].value_counts().head(5) if hasattr(self, 'sliding_merged') else pd.Series()
                heap_top_tickers = self.heap_merged['ticker'].value_counts().head(5) if hasattr(self, 'heap_merged') else pd.Series()
                
                date_range = "N/A"
                if 'date' in self.features_df.columns:
                    date_range = f"{self.features_df['date'].min()} to {self.features_df['date'].max()}"
                
                report = f"""
STOCK MARKET ANOMALY DETECTION REPORT
=====================================

DATA OVERVIEW:
- Total data points: {len(self.features_df):,}
- Date range: {date_range}
- Unique tickers: {self.features_df['ticker'].nunique() if 'ticker' in self.features_df.columns else 'N/A'}

DETECTION RESULTS:
- Sliding Window anomalies: {len(self.sliding_df):,} ({sliding_rate:.2f}%)
- Heap-Based anomalies: {len(self.heap_df):,} ({heap_rate:.2f}%)
- Overlapping anomalies: {overlap_count:,} ({overlap_rate:.2f}%)

DETECTION QUALITY ASSESSMENT:
✅ Sliding Window: {'Excellent' if 2 <= sliding_rate <= 5 else 'Good' if sliding_rate < 10 else 'High'} detection rate ({sliding_rate:.1f}%)
{'✅' if heap_rate < 10 else '⚠️'} Heap-Based: {'Good' if heap_rate < 10 else 'High'} detection rate ({heap_rate:.1f}%)

TOP AFFECTED TICKERS (Sliding Window):
{sliding_top_tickers.to_string() if len(sliding_top_tickers) > 0 else 'No data available'}

TOP AFFECTED TICKERS (Heap-Based):
{heap_top_tickers.to_string() if len(heap_top_tickers) > 0 else 'No data available'}

RECOMMENDATIONS:
- {'Heap algorithm tuning successful' if heap_rate < 10 else 'Consider further tuning heap algorithm thresholds'}
- Monitor tickers with high anomaly counts for potential issues
- {'Both methods show good agreement' if overlap_rate > 2 else 'Consider investigating method differences'}
"""

                with open(self.plots_dir / "improved_anomaly_summary.txt", "w", encoding="utf-8") as f:
                    f.write(report)

                print("✅ Summary report created")
            else:
                print("⚠️ Could not create summary report - missing anomaly flag data")
                
        except Exception as e:
            print(f"Error creating summary report: {e}")
    
    def run_full_analysis(self):
        """Run the complete analysis and visualization"""
        print("🚀 Starting anomaly analysis...")
        
        self.create_overview_plots()
        self.create_ticker_specific_plots()
        self.create_summary_report()
        
        print("\n" + "="*50)
        print("📊 ANALYSIS COMPLETE")
        print("="*50)
        print(f"📈 Total sliding window anomalies: {len(self.sliding_df):,}")
        print(f"📈 Total heap-based anomalies: {len(self.heap_df):,}")
        print(f"📁 Enhanced plots saved in: {self.plots_dir}/")
        print(f"📄 Detailed summary saved in: {self.plots_dir}/improved_anomaly_summary.txt")
        print("="*50)

if __name__ == "__main__":
    # Initialize and run the improved visualizer
    visualizer = ImprovedAnomalyVisualizer()
    visualizer.run_full_analysis()