import os
import json
from datetime import datetime
import numpy as np


def read_single_file_lead_data(file_path, target_lead=4, total_leads=9):
    """
    Read single raw ECG file's selected lead data and timestamps (original logic unchanged).

    Parameters:
        file_path (str): Path to single ECG text file.
        target_lead (int): Selected lead number (1-based).
        total_leads (int): Total number of leads in data.

    Returns:
        tuple: (file_signal_segments, file_timestamps)
    """
    file_signal_segments = []
    file_timestamps = []
    lead_index = target_lead - 1

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_str in f:
                try:
                    data_line = json.loads(line_str)
                    record_time = datetime.fromtimestamp(data_line['recordTime'])
                    file_timestamps.append(record_time)

                    if lead_index < 0 or lead_index >= total_leads:
                        raise IndexError(f"Lead {target_lead} out of range (1-{total_leads})")

                    wave_data = data_line['data']['waveDataList'][lead_index]['waveDataVoList']
                    signal_values = [float(item['sample']) for item in wave_data]
                    file_signal_segments.append(signal_values)

                except json.JSONDecodeError as e:
                    print(f'  Warning: JSON decode error in {os.path.basename(file_path)}: {e}. Skipped line.')
                except (KeyError, IndexError) as e:
                    print(f'  Warning: Data error in {os.path.basename(file_path)}: {e}. Skipped line.')
                except Exception as e:
                    print(f'  Warning: Unexpected error in {os.path.basename(file_path)}: {e}. Skipped line.')

    except Exception as e:
        print(f'Error reading file {os.path.basename(file_path)}: {e}. Skipped file.')
        return [], []

    print(f'Processed file: {os.path.basename(file_path)}')
    print(f'  Extracted {len(file_signal_segments)} valid signal segments')
    return file_signal_segments, file_timestamps


def get_ecg_file_list(folder_path):
    """Get sorted list of raw ECG txt files (original logic unchanged)."""
    ecg_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            ecg_files.append(os.path.join(folder_path, filename))
    print(f"Found {len(ecg_files)} raw ECG files in directory")
    return ecg_files


def read_hr_json_file(file_path):
    """
    Read JSON format heart rate file (timestamp + HR value).

    Parameters:
        file_path (str): Path to JSON HR file.

    Returns:
        tuple: (timestamps, heart_rates)
    """
    timestamps = []
    heart_rates = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Standard JSON structure: {"timestamps": [...], "heart_rates_bpm": [...]}
            if "timestamps" in data and "heart_rates_bpm" in data:
                for ts_str, hr in zip(data["timestamps"], data["heart_rates_bpm"]):
                    try:
                        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').timestamp()
                        timestamps.append(ts)
                        heart_rates.append(float(hr))
                    except Exception as e:
                        print(f'  Warning: Invalid data in {os.path.basename(file_path)}: {e}. Skipped entry.')
            else:
                print(f'  Warning: Invalid JSON structure in {os.path.basename(file_path)}. Skipped file.')
    except Exception as e:
        print(f'Error reading JSON file {os.path.basename(file_path)}: {e}. Skipped file.')
        return [], []

    print(f'Processed JSON HR file: {os.path.basename(file_path)}')
    print(f'  Extracted {len(heart_rates)} valid HR data points')
    return timestamps, heart_rates


def get_hr_json_file_list(folder_path):
    """Get sorted list of JSON heart rate files."""
    json_files = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            json_files.append(os.path.join(folder_path, filename))
    print(f"Found {len(json_files)} JSON HR files in directory")
    return json_files