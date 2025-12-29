import numpy as np
import matplotlib.pyplot as plt


def plot_hr_histogram(all_heart_rates, save_path=None):
    """
    Plot HR frequency distribution histogram (original logic unchanged).

    Parameters:
        all_heart_rates (list): Combined HR values from all files.
        save_path (str): Path to save plot (None = don't save).

    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    if not all_heart_rates:
        raise ValueError("No valid heart rate data to plot")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    counts, bins, patches = ax.hist(all_heart_rates, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    mean_hr = np.mean(all_heart_rates)
    ax.axvline(mean_hr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_hr:.1f} BPM')
    median_hr = np.median(all_heart_rates)
    ax.axvline(median_hr, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_hr:.1f} BPM')

    ax.set_title('Heart Rate Frequency Distribution', fontsize=12, pad=10)
    ax.set_xlabel('Heart Rate (BPM)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(alpha=0.5, linestyle='--', axis='y')
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HR Histogram saved to: {save_path}")

    return fig


def plot_hr_poincare(all_heart_rates, save_path=None):
    """
    Plot HR Poincare scatter plot (HR(n) vs HR(n+1)).

    Parameters:
        all_heart_rates (list): Combined HR values from all files.
        save_path (str): Path to save plot (None = don't save).

    Returns:
        matplotlib.figure.Figure: Generated figure object.
    """
    if len(all_heart_rates) < 2:
        raise ValueError("At least 2 HR data points required for Poincare plot")

    hr_array = np.array(all_heart_rates)
    hr_n = hr_array[:-1]  # HR(n)
    hr_n1 = hr_array[1:]  # HR(n+1)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.scatter(hr_n, hr_n1, color='darkgreen', s=10, alpha=0.6, label='HR(n) vs HR(n+1)')

    # Add identity line (HR(n) = HR(n+1))
    min_hr = min(min(hr_n), min(hr_n1))
    max_hr = max(max(hr_n), max(hr_n1))
    ax.plot([min_hr, max_hr], [min_hr, max_hr], 'r--', linewidth=2, label='Identity Line (HR(n)=HR(n+1))')

    ax.set_title('HR Poincare Scatter Plot', fontsize=12, pad=10)
    ax.set_xlabel('HR(n) (BPM)', fontsize=10)
    ax.set_ylabel('HR(n+1) (BPM)', fontsize=10)
    ax.grid(alpha=0.5, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HR Poincare plot saved to: {save_path}")

    return fig