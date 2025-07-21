# trigger_save_all.py
from metavision_core.event_io.raw_reader import RawReader
import numpy as np

# ——— 設定 ———
raw_file   = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/recording_250708_224544_288.raw"
output_npy = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/triggers.npy"

# ——— 一括読み込み ———
reader = RawReader(raw_file)
print(f"Opening RAW file: {raw_file}")

# 非現実的に大きな時間ウィンドウを指定して、一度で全イベントを読み込む
# （ファイル長に応じて十分大きな値を指定してください）
reader.load_delta_t(10**15)

# 全イベントを取得
all_triggers = reader.get_ext_trigger_events()

# ——— 結果を .npy に保存 ———
np.save(output_npy, all_triggers)
print(f"Saved {len(all_triggers)} trigger events to '{output_npy}'")

# ——— (オプション) 確認表示 ———
print("Fields:", all_triggers.dtype.names)
print("First 5 events:", all_triggers[:5])
