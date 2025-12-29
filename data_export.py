import json
import os
from datetime import datetime


def export_to_json(combined_timestamps, combined_heart_rates, hr_stats=None, export_path="ecg_hr_results.json"):
    """
    Export analysis results + time-domain stats to JSON file (English parameters).

    Parameters:
        combined_timestamps (list): Combined timestamps from all files.
        combined_heart_rates (list): Combined heart rates from all files.
        hr_stats (dict, optional): HR time-domain statistics. Defaults to None.
        export_path (str): Path to save JSON file.

    Returns:
        bool: True if export successful, False otherwise.
    """
    try:
        readable_timestamps = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                               for ts in combined_timestamps]

        export_data = {
            "analysis_info": {
                "export_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_hr_data_points": len(combined_heart_rates)
            },
            "heart_rate_time_domain": {
                "timestamps": readable_timestamps,
                "heart_rates_bpm": [round(hr, 2) for hr in combined_heart_rates]
            }
        }

        # 新增：导出HR时域统计量
        if hr_stats and isinstance(hr_stats, dict):
            export_data["hr_time_domain_stats"] = hr_stats

        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"Data exported to: {export_path}")
        return True

    except Exception as e:
        print(f"Error exporting JSON: {str(e)}")
        return False