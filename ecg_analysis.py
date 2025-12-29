import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time  # 新增：用于暂停逻辑


# 【原有函数：analyze_single_file_hr、plot_combined_hr、plot_hr_time_line、create_plot_window 保持不变】
def analyze_single_file_hr(signal_segments, timestamps, sampling_rate=250):
    """
    Analyze heart rate for a single file (minute-wise, original logic unchanged).

    Parameters:
        signal_segments (list): List of signal segments from single file.
        timestamps (list): List of timestamps corresponding to each segment.
        sampling_rate (int): Sampling rate in Hz.

    Returns:
        tuple: (file_timestamps, file_heart_rates)
    """
    if not signal_segments or not timestamps:
        return [], []

    minute_signals = {}  # Key: (year, month, day, hour, minute), Value: merged signal
    for segment, seg_time in zip(signal_segments, timestamps):
        minute_key = (seg_time.year, seg_time.month, seg_time.day, seg_time.hour, seg_time.minute)
        if minute_key not in minute_signals:
            minute_signals[minute_key] = []
        minute_signals[minute_key].extend(segment)

    file_timestamps = []
    file_heart_rates = []

    for minute_key in sorted(minute_signals.keys()):
        merged_signal = np.array(minute_signals[minute_key])
        if len(merged_signal) < sampling_rate * 5:
            continue

        try:
            avg_signal = np.mean(merged_signal)
            std_signal = np.std(merged_signal)
            threshold = avg_signal + 1.5 * std_signal
            peaks = np.where(merged_signal > threshold)[0]

            min_distance = int(sampling_rate * 0.2)
            filtered_peaks = []
            for peak in peaks:
                if not filtered_peaks or peak - filtered_peaks[-1] > min_distance:
                    filtered_peaks.append(peak)

            if len(filtered_peaks) < 2:
                continue

            rr_intervals = np.diff(filtered_peaks) / sampling_rate
            heart_rates = 60 / rr_intervals
            minute_avg_hr = np.mean(heart_rates)

            if np.isnan(minute_avg_hr) or minute_avg_hr < 30 or minute_avg_hr > 200:
                continue

            minute_time = datetime(*minute_key) + timedelta(minutes=1)
            file_timestamps.append(minute_time.timestamp())
            file_heart_rates.append(minute_avg_hr)

        except Exception as e:
            continue

    return file_timestamps, file_heart_rates


def plot_combined_hr(all_timestamps, all_heart_rates, save_path=None):
    """
    Plot HR vs Time scatter plot (main UI plot, higher transparency).

    Parameters:
        all_timestamps (list): Combined timestamps from all files.
        all_heart_rates (list): Combined HR values from all files.
        save_path (str): Path to save plot (None = don't save).

    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    if not all_timestamps or not all_heart_rates:
        raise ValueError("No valid heart rate data to plot")

    times = [datetime.fromtimestamp(ts) for ts in all_timestamps]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    # Increase transparency (alpha=0.4)
    ax.scatter(times, all_heart_rates, color='crimson', s=15, alpha=0.4, label='Minute Avg HR')
    global_avg = np.mean(all_heart_rates)
    ax.axhline(y=global_avg, color='navy', linestyle='--', linewidth=2, label=f'Global Avg: {global_avg:.1f} BPM')

    ax.set_title('Heart Rate vs Time (Scatter Plot)', fontsize=12, pad=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Heart Rate (BPM)', fontsize=10)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    ax.grid(alpha=0.5, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HR Scatter plot saved to: {save_path}")

    return fig


def plot_hr_time_line(all_timestamps, all_heart_rates, save_path=None):
    """
    Plot HR vs Time line plot (new chart).

    Parameters:
        all_timestamps (list): Combined timestamps from all files.
        all_heart_rates (list): Combined HR values from all files.
        save_path (str): Path to save plot (None = don't save).

    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    if not all_timestamps or not all_heart_rates:
        raise ValueError("No valid heart rate data to plot")

    times = [datetime.fromtimestamp(ts) for ts in all_timestamps]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    ax.plot(times, all_heart_rates, color='darkblue', linewidth=1.5, alpha=0.8, label='Minute Avg HR')
    global_avg = np.mean(all_heart_rates)
    ax.axhline(y=global_avg, color='red', linestyle='--', linewidth=2, label=f'Global Avg: {global_avg:.1f} BPM')

    ax.set_title('Heart Rate vs Time (Line Plot)', fontsize=12, pad=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Heart Rate (BPM)', fontsize=10)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    ax.grid(alpha=0.5, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HR Line plot saved to: {save_path}")

    return fig


def create_plot_window(fig, title):
    """
    Create a new Tkinter window with plot and navigation toolbar (zoom/pan).

    Parameters:
        fig (matplotlib.figure.Figure): Plot figure.
        title (str): Window title.
    """
    window = tk.Toplevel()
    window.title(title)
    window.geometry("1000x600")

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    # Add navigation toolbar (supports zoom/pan/save)
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()

    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return window


# 【新增：HR时域统计量计算函数】
def calculate_hr_time_domain_stats(heart_rates):
    """
    Calculate HR time-domain statistical metrics (all English parameters).

    Parameters:
        heart_rates (list/np.array): Combined heart rate values.

    Returns:
        dict: Time-domain stats with English keys.
    """
    if not heart_rates or len(heart_rates) < 2:
        return {}

    hr_array = np.array(heart_rates)
    hr_diff = np.diff(hr_array)  # 相邻HR差值

    # 核心统计量（全部英文命名）
    stats = {
        "mean_hr": np.mean(hr_array),  # 均值
        "median_hr": np.median(hr_array),  # 中位数
        "std_hr": np.std(hr_array, ddof=1),  # 标准差 (SDNN 等价)
        "var_hr": np.var(hr_array, ddof=1),  # 方差
        "min_hr": np.min(hr_array),  # 最小值
        "max_hr": np.max(hr_array),  # 最大值
        "range_hr": np.max(hr_array) - np.min(hr_array),  # 极差
        "rmssd": np.sqrt(np.mean(np.square(hr_diff))),  # 相邻HR差值均方根
        "cv_sd": np.sqrt(np.mean(np.square(hr_diff))) / np.mean(hr_array),  # CVSD (RMSSD/均值)
        "cv_nn": np.std(hr_array, ddof=1) / np.mean(hr_array),  # CVNN (SDNN/均值)
        "pnn50": 100 * np.sum(np.abs(hr_diff) > 5) / len(hr_diff)  # 相邻HR差值>5BPM的占比
    }

    # 保留2位小数
    for key in stats:
        stats[key] = round(stats[key], 2)

    return stats


# Need to import tkinter here (since used in create_plot_window)
import tkinter as tk