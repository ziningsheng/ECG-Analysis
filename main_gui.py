import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime, timedelta
import numpy as np
import time

# Import custom modules
from data_read import (
    get_ecg_file_list, read_single_file_lead_data,
    get_hr_json_file_list, read_hr_json_file
)
from ecg_analysis import (
    analyze_single_file_hr, plot_combined_hr, plot_hr_time_line,
    create_plot_window, calculate_hr_time_domain_stats
)
from hrv_analysis import plot_hr_histogram, plot_hr_poincare
from data_export import export_to_json


class ECGHeartRateGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECG Heart Rate Analyzer (HRV Enhanced)")
        self.geometry("1600x900")
        self.resizable(True, True)

        # Initialize variables (all English parameters)
        self.folder_path = tk.StringVar()
        self.total_leads = tk.IntVar(value=9)
        self.sampling_rate = tk.IntVar(value=250)
        self.target_lead = tk.IntVar(value=4)
        self.input_type = tk.StringVar(value="raw_ecg")
        self.is_analyzing = False
        self.is_paused = tk.BooleanVar(value=False)

        # Store analysis results
        self.combined_timestamps = []  # 存储timestamp（数值型）
        self.combined_heart_rates = []
        self.hr_global_stats = {}  # 全局统计量
        self.hr_range_stats = {}  # 时间段统计量

        # Create UI widgets
        self.create_widgets()

    def create_widgets(self):
        # 1. Top control frame
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(fill='x', anchor='n')

        # Folder selection
        ttk.Label(control_frame, text="Data Folder:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(control_frame, textvariable=self.folder_path, width=50).grid(row=0, column=1, padx=5, pady=5,
                                                                               sticky='ew')
        ttk.Button(control_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=5, pady=5)

        # Input type selection
        input_frame = ttk.LabelFrame(control_frame, text="Input Type", padding="5")
        input_frame.grid(row=0, column=3, padx=10, pady=5, sticky='w')
        ttk.Radiobutton(input_frame, text="Raw ECG Files (.txt)", variable=self.input_type,
                        value="raw_ecg", command=self.update_param_visibility).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(input_frame, text="HR JSON Files (.json)", variable=self.input_type,
                        value="hr_json", command=self.update_param_visibility).grid(row=0, column=1, padx=5)

        # Parameters frame (English labels)
        self.param_frame = ttk.LabelFrame(control_frame, text="ECG Parameters", padding="10")
        self.param_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='ew')

        ttk.Label(self.param_frame, text="Total ECG Leads:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.total_leads_entry = ttk.Entry(self.param_frame, textvariable=self.total_leads, width=10)
        self.total_leads_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.param_frame, text="Target Lead:").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.lead_combobox = ttk.Combobox(self.param_frame, textvariable=self.target_lead, width=8, state="readonly")
        self.update_lead_options()
        self.lead_combobox.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(self.param_frame, text="Sampling Rate (Hz):").grid(row=0, column=4, padx=10, pady=5, sticky='w')
        self.sampling_rate_entry = ttk.Entry(self.param_frame, textvariable=self.sampling_rate, width=10)
        self.sampling_rate_entry.grid(row=0, column=5, padx=5, pady=5)

        # Operation buttons (新增时间段统计按钮)
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=10, sticky='w')

        self.analyze_btn = ttk.Button(btn_frame, text="Start Analysis", command=self.start_analysis)
        self.analyze_btn.grid(row=0, column=0, padx=5)

        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.grid(row=0, column=2, padx=5)

        # 新增：时间段统计按钮（初始禁用）
        self.range_stats_btn = ttk.Button(btn_frame, text="Calculate Time-Range Stats",
                                          command=self.open_time_range_dialog, state='disabled')
        self.range_stats_btn.grid(row=0, column=3, padx=5)

        self.export_btn = ttk.Button(btn_frame, text="Export JSON", command=self.export_data, state='disabled')
        self.export_btn.grid(row=0, column=4, padx=5)

        # Chart buttons
        self.line_plot_btn = ttk.Button(btn_frame, text="Show HR Line Plot", command=self.show_hr_line_plot,
                                        state='disabled')
        self.line_plot_btn.grid(row=0, column=5, padx=5)

        self.hist_plot_btn = ttk.Button(btn_frame, text="Show HR Histogram", command=self.show_hr_histogram,
                                        state='disabled')
        self.hist_plot_btn.grid(row=0, column=6, padx=5)

        self.poincare_btn = ttk.Button(btn_frame, text="Show Poincare Plot", command=self.show_poincare_plot,
                                       state='disabled')
        self.poincare_btn.grid(row=0, column=7, padx=5)

        # 2. Content frame (log + stats + main plot)
        content_frame = ttk.Frame(self, padding="10")
        content_frame.pack(fill='both', expand=True)

        # Left column: Log + Global Stats + Range Stats
        left_col = ttk.Frame(content_frame)
        left_col.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # Log area
        log_frame = ttk.LabelFrame(left_col, text="Analysis Log", padding="10")
        log_frame.pack(fill='both', expand=True, padx=0, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, width=50, height=15, font=("Arial", 10))
        self.log_text.pack(fill='both', expand=True)

        # Global HR stats (原有全局统计量)
        global_stats_frame = ttk.LabelFrame(left_col, text="Global HR Time-Domain Stats", padding="10")
        global_stats_frame.pack(fill='both', expand=True, padx=0, pady=5)
        self.global_stats_labels = self._create_stats_labels(global_stats_frame)

        # Time-range HR stats (新增时间段统计量)
        range_stats_frame = ttk.LabelFrame(left_col, text="Time-Range HR Time-Domain Stats", padding="10")
        range_stats_frame.pack(fill='both', expand=True, padx=0, pady=5)
        self.range_stats_labels = self._create_stats_labels(range_stats_frame)

        # Right column: Main plot (scatter plot)
        self.hr_scatter_frame = ttk.LabelFrame(content_frame, text="Heart Rate vs Time (Scatter Plot)", padding="10")
        self.hr_scatter_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        # Grid configuration
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=3)
        content_frame.grid_rowconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        self.param_frame.grid_columnconfigure(5, weight=1)

    def _create_stats_labels(self, parent_frame):
        """Helper: Create standardized stats labels grid (return label dict)."""
        stats_keys = [
            "mean_hr", "median_hr", "std_hr", "var_hr",
            "min_hr", "max_hr", "range_hr", "rmssd",
            "cv_sd", "cv_nn", "pnn50"
        ]
        stats_names = [
            "Mean HR (BPM):", "Median HR (BPM):", "SDNN (BPM):", "Variance (BPM²):",
            "Min HR (BPM):", "Max HR (BPM):", "HR Range (BPM):", "RMSSD (BPM):",
            "CVSD:", "CVNN:", "PNN50 (%) :"
        ]
        labels = {}
        for idx, (key, name) in enumerate(zip(stats_keys, stats_names)):
            row = idx // 2
            col = idx % 2
            ttk.Label(parent_frame, text=name, font=("Arial", 9, "bold")).grid(row=row, column=2 * col, padx=5, pady=3,
                                                                               sticky='w')
            lbl = ttk.Label(parent_frame, text="-", font=("Arial", 9))
            lbl.grid(row=row, column=2 * col + 1, padx=5, pady=3, sticky='w')
            labels[key] = lbl
        return labels

    def update_param_visibility(self):
        """Show/hide ECG parameters based on input type."""
        if self.input_type.get() == "raw_ecg":
            self.total_leads_entry.config(state='normal')
            self.lead_combobox.config(state='readonly')
            self.sampling_rate_entry.config(state='normal')
        else:
            self.total_leads_entry.config(state='disabled')
            self.lead_combobox.config(state='disabled')
            self.sampling_rate_entry.config(state='disabled')

    def update_lead_options(self):
        """Update lead combobox options based on total leads."""
        try:
            total = max(int(self.total_leads.get()), 1)
            self.lead_combobox['values'] = [str(i) for i in range(1, total + 1)]
            if self.target_lead.get() < 1 or self.target_lead.get() > total:
                self.target_lead.set(1)
        except ValueError:
            self.lead_combobox['values'] = [str(i) for i in range(1, 10)]
            self.target_lead.set(1)

    def browse_folder(self):
        """Open folder dialog to select data folder."""
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.folder_path.set(folder)
            self.log(f"Selected data folder: {folder}")

    def log(self, message):
        """Thread-safe log writing."""
        self.after(0, lambda: self._log_in_mainthread(message))

    def _log_in_mainthread(self, message):
        """Write log to text widget (main thread only)."""
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def clear_results(self):
        """Clear all analysis results and plots."""
        self.combined_timestamps = []
        self.combined_heart_rates = []
        self.hr_global_stats = {}
        self.hr_range_stats = {}
        self.is_paused.set(False)

        # Clear plots
        for widget in self.hr_scatter_frame.winfo_children():
            widget.destroy()

        # Clear stats labels
        for lbl in self.global_stats_labels.values():
            lbl.config(text="-")
        for lbl in self.range_stats_labels.values():
            lbl.config(text="-")

        # Disable buttons
        self.export_btn.config(state='disabled')
        self.line_plot_btn.config(state='disabled')
        self.hist_plot_btn.config(state='disabled')
        self.poincare_btn.config(state='disabled')
        self.pause_btn.config(state='disabled', text="Pause")
        self.range_stats_btn.config(state='disabled')  # 禁用时间段统计按钮

        self.log("Results cleared successfully.")

    def toggle_pause(self):
        """Toggle pause/resume state during analysis."""
        if self.is_paused.get():
            self.is_paused.set(False)
            self.pause_btn.config(text="Pause")
            self.log("Analysis resumed.")
        else:
            self.is_paused.set(True)
            self.pause_btn.config(text="Resume")
            self.log("Analysis paused (click Resume to continue).")

    def start_analysis(self):
        """Start analysis (threaded to avoid UI freeze)."""
        if self.is_analyzing:
            return
        if not self.folder_path.get():
            self.log("Error: Please select a data folder first!")
            return

        # Validate parameters
        try:
            if self.input_type.get() == "raw_ecg":
                total_leads = int(self.total_leads.get())
                sampling_rate = int(self.sampling_rate.get())
                target_lead = self.target_lead.get()
                if total_leads <= 0 or sampling_rate <= 0 or not (1 <= target_lead <= total_leads):
                    raise ValueError("Invalid ECG parameters (positive integers required)")
            else:
                total_leads = 0
                sampling_rate = 0
                target_lead = 0
        except ValueError as e:
            self.log(f"Error: {str(e)}. Please check input parameters.")
            return

        # Update UI state
        self.is_analyzing = True
        self.analyze_btn.config(text="Analyzing...", state='disabled')
        self.pause_btn.config(state='normal')
        self.clear_btn.config(state='disabled')
        self.clear_results()

        # Log start
        self.log("\n" + "=" * 60)
        self.log(f"Starting analysis (Input Type: {self.input_type.get().upper()})")
        if self.input_type.get() == "raw_ecg":
            self.log(f"- Total Leads: {total_leads}, Target Lead: {target_lead}")
            self.log(f"- Sampling Rate: {sampling_rate} Hz")
        self.log("=" * 60)

        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(self.folder_path.get(), total_leads, target_lead, sampling_rate)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis(self, folder_path, total_leads, target_lead, sampling_rate):
        """Core analysis logic (background thread, with pause support)."""
        try:
            if self.input_type.get() == "raw_ecg":
                # Process raw ECG files
                ecg_files = get_ecg_file_list(folder_path)
                if not ecg_files:
                    self.log("Error: No raw ECG (.txt) files found in selected folder!")
                    return

                total_files = len(ecg_files)
                self.log(f"Found {total_files} raw ECG files. Starting processing...")

                for idx, file_path in enumerate(ecg_files, 1):
                    # Check pause state
                    while self.is_paused.get():
                        time.sleep(0.5)

                    filename = os.path.basename(file_path)
                    self.log(f"\nProcessing file {idx}/{total_files}: {filename}")

                    # Read signal data
                    signal_segments, timestamps = read_single_file_lead_data(file_path, target_lead, total_leads)
                    if not signal_segments:
                        self.log(f"  Warning: No valid signal data in {filename}. Skipping.")
                        continue

                    # Analyze HR
                    file_ts, file_hr = analyze_single_file_hr(signal_segments, timestamps, sampling_rate)
                    if file_ts and file_hr:
                        self.combined_timestamps.extend(file_ts)
                        self.combined_heart_rates.extend(file_hr)
                        self.log(f"  Success: Extracted {len(file_hr)} valid HR points.")
                    else:
                        self.log(f"  Warning: No valid HR data from {filename}.")

            else:
                # Process HR JSON files
                json_files = get_hr_json_file_list(folder_path)
                if not json_files:
                    self.log("Error: No HR JSON (.json) files found in selected folder!")
                    return

                total_files = len(json_files)
                self.log(f"Found {total_files} HR JSON files. Starting processing...")

                for idx, file_path in enumerate(json_files, 1):
                    # Check pause state
                    while self.is_paused.get():
                        time.sleep(0.5)

                    filename = os.path.basename(file_path)
                    self.log(f"\nProcessing file {idx}/{total_files}: {filename}")

                    # Read JSON HR data
                    file_ts, file_hr = read_hr_json_file(file_path)
                    if file_ts and file_hr:
                        self.combined_timestamps.extend(file_ts)
                        self.combined_heart_rates.extend(file_hr)
                        self.log(f"  Success: Extracted {len(file_hr)} valid HR points.")
                    else:
                        self.log(f"  Warning: No valid HR data from {filename}.")

            # Check pause state after processing all files
            while self.is_paused.get():
                time.sleep(0.5)

            # Analysis summary
            self.log("\n" + "=" * 60)
            self.log("Analysis completed!")
            self.log(f"Total valid HR data points: {len(self.combined_heart_rates)}")

            if self.combined_heart_rates:
                # Calculate global stats
                self.hr_global_stats = calculate_hr_time_domain_stats(self.combined_heart_rates)
                self.log(f"Global average HR: {self.hr_global_stats['mean_hr']:.1f} BPM")
                self.log(f"HR range: {self.hr_global_stats['min_hr']:.1f} - {self.hr_global_stats['max_hr']:.1f} BPM")

                # Update UI: plot + global stats + enable range stats button
                self.after(0, self.display_scatter_plot)
                self.after(0, self.display_global_stats)
                self.after(0, lambda: self.range_stats_btn.config(state='normal'))  # 启用时间段统计按钮

                # Enable other buttons
                self.after(0, lambda: self.export_btn.config(state='normal'))
                self.after(0, lambda: self.line_plot_btn.config(state='normal'))
                self.after(0, lambda: self.hist_plot_btn.config(state='normal'))
                self.after(0, lambda: self.poincare_btn.config(state='normal'))
            else:
                self.log("Warning: No valid HR data found in any file.")

        except Exception as e:
            self.log(f"\nError during analysis: {str(e)}")
        finally:
            # Restore UI state
            self.after(0, lambda: self.analyze_btn.config(text="Start Analysis", state='normal'))
            self.after(0, lambda: self.pause_btn.config(state='disabled', text="Pause"))
            self.after(0, lambda: self.clear_btn.config(state='normal'))
            self.is_analyzing = False
            self.is_paused.set(False)

    def display_scatter_plot(self):
        """Display HR scatter plot in main UI (higher transparency)."""
        for widget in self.hr_scatter_frame.winfo_children():
            widget.destroy()

        try:
            fig = plot_combined_hr(self.combined_timestamps, self.combined_heart_rates)
            canvas = FigureCanvasTkAgg(fig, master=self.hr_scatter_frame)
            canvas.draw()

            # Add navigation toolbar (zoom/pan for main plot)
            toolbar = NavigationToolbar2Tk(canvas, self.hr_scatter_frame)
            toolbar.update()

            canvas.get_tk_widget().pack(fill='both', expand=True)
        except Exception as e:
            self.log(f"Error displaying scatter plot: {str(e)}")

    def display_global_stats(self):
        """Update global stats labels (main thread)."""
        if not self.hr_global_stats:
            return
        for key, lbl in self.global_stats_labels.items():
            if key in self.hr_global_stats:
                lbl.config(text=str(self.hr_global_stats[key]))

    def display_range_stats(self):
        """Update time-range stats labels (main thread)."""
        if not self.hr_range_stats:
            return
        for key, lbl in self.range_stats_labels.items():
            if key in self.hr_range_stats:
                lbl.config(text=str(self.hr_range_stats[key]))

    # 新增：时间段选择弹窗
    def open_time_range_dialog(self):
        """Open dialog to select time range for stats calculation."""
        if not self.combined_timestamps:
            self.log("Error: No HR data available for time-range stats!")
            return

        # Convert timestamps to readable format for reference
        min_ts = min(self.combined_timestamps)
        max_ts = max(self.combined_timestamps)
        min_time_str = datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d %H:%M:%S')
        max_time_str = datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d %H:%M:%S')

        # Create dialog window
        dialog = tk.Toplevel(self)
        dialog.title("Select Time Range for HR Stats")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        dialog.transient(self)  # Attach to main window
        dialog.grab_set()  # Modal dialog

        # Dialog content
        ttk.Label(dialog, text=f"Data Time Range: {min_time_str} ~ {max_time_str}",
                  font=("Arial", 9)).pack(pady=10)

        # Start time input
        start_frame = ttk.Frame(dialog)
        start_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(start_frame, text="Start Time (YYYY-MM-DD HH:MM:SS):", width=25).grid(row=0, column=0, sticky='w')
        start_entry = ttk.Entry(start_frame, width=20)
        start_entry.grid(row=0, column=1, padx=5)
        start_entry.insert(0, min_time_str)  # Default: min time

        # End time input
        end_frame = ttk.Frame(dialog)
        end_frame.pack(fill='x', padx=20, pady=5)
        ttk.Label(end_frame, text="End Time (YYYY-MM-DD HH:MM:SS):", width=25).grid(row=0, column=0, sticky='w')
        end_entry = ttk.Entry(end_frame, width=20)
        end_entry.grid(row=0, column=1, padx=5)
        end_entry.insert(0, max_time_str)  # Default: max time

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        def calculate_range_stats():
            """Calculate stats for selected time range."""
            try:
                # Parse input time strings
                start_str = start_entry.get().strip()
                end_str = end_entry.get().strip()
                start_ts = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S').timestamp()
                end_ts = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S').timestamp()

                if start_ts > end_ts:
                    self.log("Error: Start time cannot be later than end time!")
                    return
                if start_ts < min_ts or end_ts > max_ts:
                    self.log(f"Warning: Time range exceeds data range ({min_time_str} ~ {max_time_str})")

                # Filter HR data by time range
                filtered_ts = []
                filtered_hr = []
                for ts, hr in zip(self.combined_timestamps, self.combined_heart_rates):
                    if start_ts <= ts <= end_ts:
                        filtered_ts.append(ts)
                        filtered_hr.append(hr)

                if not filtered_hr:
                    self.log("Error: No HR data found in selected time range!")
                    dialog.destroy()
                    return

                # Calculate range stats
                self.hr_range_stats = calculate_hr_time_domain_stats(filtered_hr)
                self.after(0, self.display_range_stats)

                # Log info
                self.log("\n" + "-" * 50)
                self.log(f"Time-Range Stats Calculated: {start_str} ~ {end_str}")
                self.log(f"Valid HR points in range: {len(filtered_hr)}")
                self.log(f"Mean HR in range: {self.hr_range_stats['mean_hr']:.1f} BPM")
                self.log("-" * 50)

                dialog.destroy()

            except ValueError as e:
                self.log(f"Error parsing time: {str(e)} (format: YYYY-MM-DD HH:MM:SS)")
            except Exception as e:
                self.log(f"Error calculating time-range stats: {str(e)}")

        ttk.Button(btn_frame, text="Calculate", command=calculate_range_stats).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=5)

    def show_hr_line_plot(self):
        """Open new window for HR line plot (with zoom/pan)."""
        if not self.combined_heart_rates:
            self.log("Error: No HR data to plot!")
            return
        try:
            fig = plot_hr_time_line(self.combined_timestamps, self.combined_heart_rates)
            create_plot_window(fig, "Heart Rate vs Time (Line Plot)")
        except Exception as e:
            self.log(f"Error generating line plot: {str(e)}")

    def show_hr_histogram(self):
        """Open new window for HR histogram (with zoom/pan)."""
        if not self.combined_heart_rates:
            self.log("Error: No HR data to plot!")
            return
        try:
            fig = plot_hr_histogram(self.combined_heart_rates)
            create_plot_window(fig, "Heart Rate Frequency Distribution")
        except Exception as e:
            self.log(f"Error generating histogram: {str(e)}")

    def show_poincare_plot(self):
        """Open new window for Poincare plot (with zoom/pan)."""
        if not self.combined_heart_rates:
            self.log("Error: No HR data to plot!")
            return
        try:
            fig = plot_hr_poincare(self.combined_heart_rates)
            create_plot_window(fig, "HR Poincare Scatter Plot")
        except Exception as e:
            self.log(f"Error generating Poincare plot: {str(e)}")

    def export_data(self):
        """Export HR data + global/range stats to JSON file."""
        if not self.combined_timestamps:
            self.log("Error: No data to export!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON File", "*.json"), ("All Files", "*.*")],
            title="Export HR Data to JSON"
        )

        if save_path:
            # Combine stats for export
            export_stats = {
                "global_stats": self.hr_global_stats,
                "time_range_stats": self.hr_range_stats
            } if self.hr_range_stats else self.hr_global_stats

            success = export_to_json(self.combined_timestamps, self.combined_heart_rates, export_stats, save_path)
            self.log(f"JSON export {'successful' if success else 'failed'}: {save_path}")


if __name__ == "__main__":
    # Set matplotlib backend for interactive plots
    plt.switch_backend('TkAgg')
    app = ECGHeartRateGUI()
    app.mainloop()