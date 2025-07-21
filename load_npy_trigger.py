# simple_load_and_print.py
import numpy as np

# .npy ファイルパス
# npy_path = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/triggers.npy"
npy_path = "/home/carrobo2024/00_drowsiness_ws/video/dynamic_objects_in_the_background/triggers.npy"

# NumPy 形式でロード（構造化配列そのまま）
triggers = np.load(npy_path, allow_pickle=False)

# そのまま表示
print(triggers)
print(len(triggers))
