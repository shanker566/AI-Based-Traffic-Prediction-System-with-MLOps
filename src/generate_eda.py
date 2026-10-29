import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_eda():
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. Load dataset
    raw_path = 'data/raw/traffic.csv'
    print(f"Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # 2. Shape, columns, info
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # 3. Datetime conversion & chronological sorting
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # Feature extraction for EDA
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.day_name()
    df['DayOfWeekNum'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['Date'] = df['DateTime'].dt.date
    
    print("\nFirst 5 rows after datetime parsing:")
    print(df.head())
    
    print("\nStatistical Summary:")
    print(df[['Vehicles', 'Hour', 'Junction']].describe())
    
    # 4. Set aesthetic plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    # Plot 1: Traffic Volume Over Time
    plt.figure(figsize=(14, 6))
    for junc in sorted(df['Junction'].unique()):
        subset = df[df['Junction'] == junc]
        # Resample daily mean for clear visualization
        daily_subset = subset.set_index('DateTime').resample('D')['Vehicles'].mean()
        plt.plot(daily_subset.index, daily_subset.values, label=f'Junction {junc}', alpha=0.85, linewidth=1.5)
    plt.title('Daily Average Traffic Volume Over Time by Junction (2015 - 2017)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Vehicle Count / Day', fontsize=12)
    plt.legend(title='Junction', frameon=True)
    plt.tight_layout()
    plt.savefig('reports/figures/traffic_over_time.png', dpi=300)
    plt.close()
    print("Saved reports/figures/traffic_over_time.png")
    
    # Plot 2: Traffic Volume by Hour
    plt.figure(figsize=(10, 5))
    hourly_avg = df.groupby('Hour')['Vehicles'].mean()
    plt.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#2b5c8f', linewidth=2.5, markersize=6)
    plt.fill_between(hourly_avg.index, hourly_avg.values, color='#2b5c8f', alpha=0.2)
    plt.title('Average Traffic Volume by Hour of Day', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Hour of Day (0:00 to 23:00)', fontsize=12)
    plt.ylabel('Average Vehicle Count', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('reports/figures/traffic_by_hour.png', dpi=300)
    plt.close()
    print("Saved reports/figures/traffic_by_hour.png")
    
    # Plot 3: Traffic Volume by Day of Week
    plt.figure(figsize=(10, 5))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_avg = df.groupby('DayOfWeek')['Vehicles'].mean().reindex(days)
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
    bars = plt.bar(day_avg.index, day_avg.values, color=colors, edgecolor='black', alpha=0.85, width=0.6)
    plt.title('Average Traffic Volume by Day of Week', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Average Vehicle Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/traffic_by_day_of_week.png', dpi=300)
    plt.close()
    print("Saved reports/figures/traffic_by_day_of_week.png")
    
    # Plot 4: Traffic Volume by Junction
    plt.figure(figsize=(8, 5))
    junc_avg = df.groupby('Junction')['Vehicles'].mean()
    plt.bar([f'Junction {j}' for j in junc_avg.index], junc_avg.values, color='#2ca02c', edgecolor='black', alpha=0.85, width=0.55)
    plt.title('Average Traffic Volume Across Junctions', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Junction', fontsize=12)
    plt.ylabel('Average Vehicle Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(junc_avg.values):
        plt.text(i, v + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/traffic_by_junction.png', dpi=300)
    plt.close()
    print("Saved reports/figures/traffic_by_junction.png")
    
    print("\nEDA figures successfully generated and saved in reports/figures/!")

if __name__ == '__main__':
    run_eda()
