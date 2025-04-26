import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import os

def parse_log_file(log_file_path):
    """Parse the log file to extract timestamps and total equity values."""
    timestamps = []
    equity_values = []
    
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                # Look for lines containing "Portfolio Status:"
                if "Portfolio Status:" in line:
                    # Get the timestamp from the previous line
                    if i >= 1:
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[i-1])
                        if timestamp_match:
                            timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            
                            # Look for Total Equity in the next few lines
                            for j in range(i+1, min(i+5, len(lines))):
                                equity_line = lines[j]
                                if "Total Equity: $" in equity_line:
                                    equity_match = re.search(r'Total Equity: \$([\d,]+\.\d+)', equity_line)
                                    if equity_match:
                                        equity = float(equity_match.group(1).replace(',', ''))
                                        timestamps.append(timestamp)
                                        equity_values.append(equity)
                                        break
                i += 1
    except FileNotFoundError:
        print(f"Error: Could not find the log file at {log_file_path}")
        print("Please make sure the file exists and the path is correct.")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return None, None
    
    if not timestamps:
        print("Warning: No equity data found in the log file.")
        return None, None
    
    print(f"Found {len(timestamps)} equity data points")
    return timestamps, equity_values

def read_equity_data(log_file):
    """Read equity data from the log file."""
    equity_data = []
    current_timestamp = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for timestamp lines
            if line.startswith('2025-'):
                try:
                    current_timestamp = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            
            # Look for Total Equity in Portfolio Status sections
            if current_timestamp and 'Total Equity: $' in line:
                try:
                    equity_str = line.split('Total Equity: $')[1].strip()
                    equity = float(equity_str.replace(',', ''))
                    equity_data.append((current_timestamp, equity))
                except (ValueError, IndexError):
                    continue
    
    return equity_data

def create_equity_graph(log_file, output_path):
    """Create and save a graph of equity over time."""
    # Read equity data
    equity_data = read_equity_data(log_file)
    
    if not equity_data:
        print("No equity data found in the log file")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
    df.set_index('timestamp', inplace=True)
    
    # Filter out data points after 20:00
    df = df[df.index.hour < 20]
    
    if len(df) == 0:
        print("No data points found before 20:00")
        return
    
    # Calculate percentage change from initial equity
    initial_equity = df['equity'].iloc[0]
    df['percent_change'] = ((df['equity'] - initial_equity) / initial_equity) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the percentage change
    plt.plot(df.index, df['percent_change'], 
             color='blue', linewidth=2, marker='o', markersize=5)
    
    # Customize the plot
    plt.title('Portfolio Performance - Percentage Change Over Time', 
             fontsize=14, pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
    
    # Calculate dynamic y-axis limits with buffer
    min_change = df['percent_change'].min()
    max_change = df['percent_change'].max()
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
    if len(df) > 0:
        significant_points.append((df.index[0], df['percent_change'].iloc[0]))
    
    # Add end point
    if len(df) > 1:
        significant_points.append((df.index[-1], df['percent_change'].iloc[-1]))
    
    # Add max and min points if they exist
    if len(df) > 0:
        max_idx = df['percent_change'].idxmax()
        min_idx = df['percent_change'].idxmin()
        
        if pd.notna(max_idx) and max_idx in df.index:
            significant_points.append((max_idx, df['percent_change'].max()))
        if pd.notna(min_idx) and min_idx in df.index:
            significant_points.append((min_idx, df['percent_change'].min()))
    
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
    print("\nPortfolio Change Summary:")
    print(f"Initial Equity: ${initial_equity:,.2f}")
    if len(df) > 0:
        print(f"Final Equity: ${df['equity'].iloc[-1]:,.2f}")
        print(f"Total Change: {df['percent_change'].iloc[-1]:+.2f}%")
        print(f"Maximum Gain: {df['percent_change'].max():+.2f}%")
        print(f"Maximum Loss: {df['percent_change'].min():+.2f}%")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the most recent log file
    log_files = [f for f in os.listdir(script_dir) if f.startswith('Paper Trading Log') and f.endswith('.txt')]
    if not log_files:
        print("No log files found")
        exit(1)
    
    # Use the most recent log file
    log_file = os.path.join(script_dir, max(log_files))
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(script_dir, f"equity_graph_{timestamp}.png")
    
    print(f"Using log file: {log_file}")
    print(f"Will save graph to: {output_file}")
    
    create_equity_graph(log_file, output_file) 