import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

class LiveTradingSimulator:
    def __init__(self, log_file):
        self.log_file = log_file
        self.equity_data = []
        self.timestamps = []
        self.current_line = 0
        self.start_time = None
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', marker='o', markersize=5)
        
    def setup_plot(self):
        """Set up the initial plot configuration."""
        self.ax.set_title('Live Trading Simulation', fontsize=14, pad=20)
        self.ax.set_xlabel('Time', fontsize=12)
        self.ax.set_ylabel('Equity ($)', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Format x-axis to show time
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        
        plt.tight_layout()
    
    def reset_simulation(self):
        """Reset the simulation to start from the beginning."""
        self.equity_data = []
        self.timestamps = []
        self.current_line = 0
        self.ax.clear()
        self.setup_plot()
        self.line, = self.ax.plot([], [], 'b-', marker='o', markersize=5)
        print("\nResetting simulation...")
    
    def read_next_line(self):
        """Read the next line from the log file."""
        with open(self.log_file, 'r') as f:
            # Skip to current line
            for _ in range(self.current_line):
                next(f)
            
            try:
                line = next(f)
                self.current_line += 1
                return line
            except StopIteration:
                return None
    
    def parse_portfolio_status(self, lines):
        """Parse portfolio status from log lines."""
        for line in lines:
            if 'Total Equity: $' in line:
                try:
                    equity_str = line.split('Total Equity: $')[1].strip()
                    return float(equity_str.replace(',', ''))
                except (ValueError, IndexError):
                    return None
        return None
    
    def update_plot(self, frame):
        """Update the plot with new data."""
        # Read next trading cycle
        lines = []
        line = self.read_next_line()
        
        while line and '=== Trading Cycle' not in line:
            lines.append(line)
            line = self.read_next_line()
        
        if not lines:
            # End of file reached, reset simulation
            self.reset_simulation()
            return self.line,
        
        # Parse timestamp and equity
        for line in lines:
            if line.startswith('2025-'):
                try:
                    timestamp = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                    equity = self.parse_portfolio_status(lines)
                    if equity is not None:
                        self.timestamps.append(timestamp)
                        self.equity_data.append(equity)
                except ValueError:
                    continue
        
        # Update plot
        if self.timestamps and self.equity_data:
            self.line.set_data(self.timestamps, self.equity_data)
            
            # Adjust x-axis limits
            self.ax.set_xlim(min(self.timestamps), max(self.timestamps))
            
            # Adjust y-axis limits with buffer
            min_value = min(self.equity_data)
            max_value = max(self.equity_data)
            buffer = (max_value - min_value) * 0.1
            self.ax.set_ylim(min_value - buffer, max_value + buffer)
            
            # Add annotations for significant points
            if len(self.equity_data) > 1:
                # Clear existing annotations
                for annotation in self.ax.texts:
                    annotation.remove()
                
                # Add new annotations
                significant_points = [
                    (self.timestamps[0], self.equity_data[0]),  # Start
                    (self.timestamps[-1], self.equity_data[-1]),  # End
                ]
                
                if len(self.equity_data) > 2:
                    max_idx = self.equity_data.index(max(self.equity_data))
                    min_idx = self.equity_data.index(min(self.equity_data))
                    significant_points.extend([
                        (self.timestamps[max_idx], self.equity_data[max_idx]),  # Max
                        (self.timestamps[min_idx], self.equity_data[min_idx])   # Min
                    ])
                
                for timestamp, equity in significant_points:
                    color = 'green' if equity >= self.equity_data[0] else 'red'
                    self.ax.annotate(f'${equity:,.2f}',
                                   (timestamp, equity),
                                   textcoords="offset points",
                                   xytext=(0,10),
                                   ha='center',
                                   color=color,
                                   fontweight='bold')
        
        return self.line,
    
    def run_simulation(self):
        """Run the live trading simulation."""
        # Initialize plot
        self.setup_plot()
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, 
                          interval=1000,  # Update every second
                          blit=True)
        
        plt.show()

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the most recent log file
    log_files = [f for f in os.listdir(script_dir) if f.startswith('Paper Trading Log') and f.endswith('.txt')]
    if not log_files:
        print("No log files found")
        return
    
    # Use the most recent log file
    log_file = os.path.join(script_dir, max(log_files))
    print(f"Using log file: {log_file}")
    
    # Create and run simulator
    simulator = LiveTradingSimulator(log_file)
    simulator.run_simulation()

if __name__ == "__main__":
    main() 