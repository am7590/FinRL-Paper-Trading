import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

def get_sp500_data(start_date, end_date):
    """Fetch S&P 500 data for the specified date range."""
    # Get S&P 500 data with 5-minute intervals
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(start=start_date, end=end_date, interval="5m")
    return data

def get_most_recent_trading_day():
    """Get the most recent trading day that has data."""
    # Try to get data from yesterday
    today = datetime.now()
    target_date = today - timedelta(days=1)
    
    # Skip weekends
    while target_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        target_date -= timedelta(days=1)
    
    # Set market hours for that day
    start_time = target_date.replace(hour=9, minute=30)
    end_time = target_date.replace(hour=20, minute=0)
    
    data = get_sp500_data(start_time, end_time)
    if len(data) > 0:
        return target_date, data
    
    return None, None

def create_sp500_graph(output_path):
    """Create and save a graph of S&P 500 percentage change over time."""
    # Get the most recent trading day with data
    trading_day, data = get_most_recent_trading_day()
    
    if trading_day is None:
        print("No trading data found for the last 5 days")
        return
    
    print(f"Using data from {trading_day.date()}")
    
    # Calculate percentage change from opening price
    opening_price = data['Open'].iloc[0]
    data['percent_change'] = ((data['Close'] - opening_price) / opening_price) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the percentage change
    plt.plot(data.index, data['percent_change'], 
             color='purple', linewidth=2, marker='o', markersize=5)
    
    # Customize the plot
    plt.title(f'S&P 500 Performance: {trading_day.date()} - Percentage Change Over Time', 
             fontsize=14, pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
    
    # Calculate dynamic y-axis limits with buffer
    min_change = data['percent_change'].min()
    max_change = data['percent_change'].max()
    buffer = max(abs(min_change), abs(max_change)) * 0.1  # 10% buffer
    
    # Set y-axis limits with buffer
    plt.ylim(min_change - buffer, max_change + buffer)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add a horizontal line at 0% for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Format the x-axis
    plt.gcf().autofmt_xdate()
    
    # Add value labels at significant points
    significant_points = []
    
    # Add start point
    if len(data) > 0:
        significant_points.append((data.index[0], data['percent_change'].iloc[0]))
    
    # Add end point
    if len(data) > 1:
        significant_points.append((data.index[-1], data['percent_change'].iloc[-1]))
    
    # Add max and min points if they exist
    if len(data) > 0:
        max_idx = data['percent_change'].idxmax()
        min_idx = data['percent_change'].idxmin()
        
        if pd.notna(max_idx) and max_idx in data.index:
            significant_points.append((max_idx, data['percent_change'].max()))
        if pd.notna(min_idx) and min_idx in data.index:
            significant_points.append((min_idx, data['percent_change'].min()))
    
    # Add annotations for significant points
    for timestamp, pct_change in significant_points:
        color = 'green' if pct_change >= 0 else 'red'
        plt.annotate(f'{pct_change:+.2f}%', 
                    (timestamp, pct_change),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    color=color,
                    fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")
    
    # Print summary statistics
    print("\nS&P 500 Change Summary:")
    print(f"Date: {trading_day.date()}")
    print(f"Opening Price: ${opening_price:,.2f}")
    if len(data) > 0:
        print(f"Closing Price: ${data['Close'].iloc[-1]:,.2f}")
        print(f"Total Change: {data['percent_change'].iloc[-1]:+.2f}%")
        print(f"Maximum Gain: {data['percent_change'].max():+.2f}%")
        print(f"Maximum Loss: {data['percent_change'].min():+.2f}%")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(script_dir, f"sp500_graph_{timestamp}.png")
    
    print(f"Will save graph to: {output_file}")
    
    create_sp500_graph(output_file) 