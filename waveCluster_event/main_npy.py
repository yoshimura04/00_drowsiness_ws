import numpy as np
import matplotlib.pyplot as plt
import pywt
from collections import deque
from scipy.ndimage import label
import os
import cv2
from tqdm import tqdm

class waveCluster:
    def __init__(self, input_npy_path, trigger_npy_path=None):
        self.input_npy_path = input_npy_path
        self.H, self.W = 720, 1280
        plt.ion()

        # ─── トリガーイベント読み込み ───
        if trigger_npy_path is not None and os.path.exists(trigger_npy_path):
            triggers = np.load(trigger_npy_path)
            print(f"Loaded {len(triggers)} trigger events from '{trigger_npy_path}'")
            # トリガー時刻だけを取り出し
            # self.trigger_times = triggers
            rise_triggers = triggers[triggers['p'] == 1]
            self.trigger_times = rise_triggers['t']
            print(f"{self.trigger_times}")
            print(f"trigger num : {len(self.trigger_times)}")
        else:
            triggers = np.zeros((0,), dtype=float)
            self.trigger_times = np.zeros((0,), dtype=float)
            print("No trigger file loaded.")
    
    def test(self,
            buffer_time=10000,
            step_time=1000,
            grid_H=360,
            grid_W=640,):

        # グリッドセルあたりのピクセル幅・高さ
        cell_h = self.H / grid_H
        cell_w = self.W / grid_W
        n_cells = grid_H * grid_W

        all_events = np.load(self.input_npy_path)
        all_events = all_events[np.argsort(all_events[:, 3])]  # 時間順に並べ替え

        # ----------初回のウィンドウを取得------------

        current_start = 0
        t_end = all_events[:, 3].max()

        events = all_events[
            (all_events[:, 3] >= current_start) &
            (all_events[:, 3] < current_start + buffer_time)
        ]
        # print(events)

        buffer_deque = deque()
        cnt_flat = np.zeros(n_cells, dtype=int)

        if events is not None and events.size > 0:
            xs = events[:, 0]
            ys = events[:, 1]
            ts = events[:, 3]

            # セルインデックス計算
            ix = np.clip((xs / cell_w).astype(int), 0, grid_W - 1)
            iy = np.clip((ys / cell_h).astype(int), 0, grid_H - 1)

            flat = iy * grid_W + ix
            
            for tsi, idx in zip(ts, flat):
                buffer_deque.append((tsi, idx))

            cnt_flat = np.bincount(flat, minlength=n_cells) # 各グリッドセルに入ったイベントの数を数える
            count_mat = cnt_flat.reshape((grid_H, grid_W))

        # -----------ウィンドウを1msずつスライド--------------

        # 総ステップ数を計算（初回ウィンドウを除く）
        total_steps = int((t_end - buffer_time) // step_time) + 1
        print(f"Total windows to process: {total_steps}")

        for step_idx in tqdm(range(total_steps), desc="Sliding windows", unit="win"):
            current_start = step_idx * step_time
            next_start = current_start + step_time

        # while current_start + buffer_time + step_time <= t_end:
        #     next_start = current_start + step_time

            # eventsの古い1ms分を消す
            while buffer_deque and buffer_deque[0][0] < next_start:
                tsi_old, idx_old = buffer_deque.popleft()
                cnt_flat[idx_old] -= 1

            # eventsに新しい1ms分を足す
            events = all_events[
                (all_events[:, 3] >= current_start + buffer_time) & 
                (all_events[:, 3] < next_start + buffer_time)
            ]
            # print(events)

            # 単一イベント時に 1D になるケースを 2D に変換
            if events.ndim == 1:
                events = events.reshape(1, -1)

            if events is not None and events.size > 0:

                xs = events[:, 0]
                ys = events[:, 1]
                ts = events[:, 3]

                ix = np.clip((xs / cell_w).astype(int), 0, grid_W - 1)
                iy = np.clip((ys / cell_h).astype(int), 0, grid_H - 1)

                flat = iy * grid_W + ix

                for tsi, idx in zip(ts, flat):
                    buffer_deque.append((tsi, idx))
                
                add_cut = np.bincount(flat, minlength=n_cells)
                cnt_flat += add_cut

            count_mat = cnt_flat.reshape((grid_H, grid_W))
            # self.plot_count_mat(count_mat)

            # -----------離散ウェーブレット変換を適用-------------
            # ・'db2' は Daubechies コアウェーブレット、level=1 は 1 段分解
            coeffs = pywt.wavedec2(
                count_mat.astype(float),
                wavelet='db2',
                mode='symmetric',
                level=1
            )

            # coeffs は [cA1, (cH1, cV1, cD1)] のリスト
            cA1, (cH1, cV1, cD1) = coeffs

            # たとえば近似係数 cA1 を可視化
            # plt.figure(figsize=(6,4))
            # plt.imshow(cA1, cmap='jet')
            # plt.title(f"Wavelet Approximation (level1) @ t={next_start}")
            # plt.colorbar(label="Coefficient")
            # plt.tight_layout()
            # # plt.show()
            # plt.pause(0.001)  # 描画を更新して0.001秒待つ
            # plt.close()       # 自動で閉じる

            # -------------しきい値でバイナリマスクを作成---------------

            # 単純に平均＋α×標準偏差で閾値を定義する例
            # mean_A = np.mean(cA1)
            # std_A  = np.std(cA1)
            # alpha = 1.0  # 調整パラメータ
            # threshold = mean_A + alpha * std_A
            threshold = 10

            # 2) バイナリマスクを作成（True がクラスタ候補領域）
            mask = cA1 > threshold

            # plt.figure(figsize=(6,4))
            # plt.imshow(mask, cmap='gray')
            # plt.title(f"Binary Mask @ t={next_start}")
            # plt.tight_layout()
            # # plt.show()
            # plt.pause(0.001)  # 描画を更新して0.001秒待つ
            # plt.close()       # 自動で閉じる

            # -------------6近傍でラベリング（step5）----------------

            structure = np.array([[0,1,0],
                                [1,1,1],
                                [0,1,0]], dtype=bool)
            labeled, num_clusters = label(mask, structure=structure)

            # print(f"検出クラスタ数: {num_clusters}")


            # -------------クラスタ重心の計算----------------
            centroids = []
            for lab in range(1, num_clusters+1):
                ys, xs = np.where(labeled == lab)
                if len(xs) == 0:
                    continue
                cy = ys.mean()
                cx = xs.mean()
                centroids.append((cy, cx))

            # --- 重心を描画して可視化 ---
            # plt.figure(figsize=(6,4))
            # plt.imshow(labeled, cmap='tab20')
            # # 各重心に赤い丸を描画
            # for (cy, cx) in centroids:
            #     plt.scatter(cx, cy, s=10, c='black', marker='o', edgecolors='white')
            # plt.title(f"Clusters & Centroids @ t={next_start}  (N={num_clusters})")
            # plt.colorbar()
            # plt.tight_layout()
            # # plt.show()
            # plt.pause(0.001)  # 描画を更新して0.001秒待つ
            # plt.close()       # 自動で閉じる

            # -------------目と口の判定（従来手法）し描写-------------------

            if 0 < num_clusters:
                # 最も近いトリガーを探す
                if self.trigger_times.size > 0:
                    # 差の絶対値が最小となるインデックス
                    trigger_idx = np.argmin(np.abs(self.trigger_times - next_start))
                    title_trigger = f"t={next_start}μs | Number of Clusters={num_clusters} | Closest Trigger: #{trigger_idx} @ {self.trigger_times[trigger_idx]}μs"
                else:
                    title_trigger = "No Triggers"

                plt.figure(figsize=(6,4))
                plt.imshow(labeled, cmap='tab20')

                for (cy, cx) in centroids:
                    plt.scatter(cx, cy, s=10, c='black', marker='o', edgecolors='white')

                if num_clusters <= 3:
                    if num_clusters == 0:
                        print(f"クラスタなし")

                    elif num_clusters == 1:
                        print(f"口を検出")
                        # 口のクラスタの重心に赤点を打つ
                        cy, cx = centroids[0]
                        plt.scatter(cx, cy, s=50, c='red', marker='o', edgecolors='white')

                    elif num_clusters == 2:
                        print(f"目を検出")
                        # 目のクラスタの重心に青点を打つ
                        for cy, cx in centroids:
                            plt.scatter(cx, cy, s=50, c='blue', marker='o', edgecolors='white')

                    else:
                        print(f"目と口を検出")
                        # 重心の y 座標で上下を判定
                        # 上側（小さい cy）が目、それ以外が口
                        centroids_sorted = sorted(centroids, key=lambda t: t[0])
                        eye_centroids = centroids_sorted[:2]
                        mouth_centroid = centroids_sorted[2]

                        # 目のクラスタの重心に青点を打つ（画像の上に位置しているクラスタ）
                        for cy, cx in eye_centroids:
                            plt.scatter(cx, cy, s=50, c='blue', marker='o', edgecolors='white')
                        # 口のクラスタの重心に赤点を打つ
                        cy, cx = mouth_centroid
                        plt.scatter(cx, cy, s=50, c='red', marker='o', edgecolors='white')

                plt.colorbar()
                # plt.title(f"t={next_start}μs | Number of Clusters={num_clusters}")
                plt.title(title_trigger, fontsize=5)
                plt.tight_layout()
                # plt.show()
                plt.pause(0.001)  # 描画を更新して0.001秒待つ

                out_dir = "output_images"
                os.makedirs(out_dir, exist_ok=True)
                filename = f"{out_dir}/frame_{next_start}us_clusters_{num_clusters}.png"
                
                plt.savefig(filename, dpi=150)
                plt.close()       # 自動で閉じる


            # current_start = next_start

    def plot_count_mat(self, count_mat):
        """
        count_mat: 2次元 NumPy 配列（grid_H x grid_W）のイベントカウント行列
        この関数を呼び出すと、画像として可視化します。
        """
        plt.figure(figsize=(6, 4))
        plt.imshow(count_mat)  # デフォルトのカラーマップを使用
        plt.title("Event Count Matrix")
        plt.colorbar(label="Count")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.tight_layout()
        # plt.show()
        plt.pause(0.001)  # 描画を更新して0.001秒待つ
        plt.close()       # 自動で閉じる
    
    def load_video_frames(self, video_path: str):
        """
        指定した動画ファイルを読み込み、全フレームをリストで返す。

        Parameters:
            video_path (str): 読み込む動画ファイルのパス

        Returns:
            frames (List[np.ndarray]): BGR フォーマットのフレーム画像のリスト
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)

        print(f"frames num : {len(self.frames)}")
        cap.release()
        return self.frames

    def show_frame(self, idx: int, figsize=(6,4)):
        """
        self.frames から指定インデックスのフレームを取り出し、
        matplotlib で表示するインスタンスメソッド。

        Parameters:
            idx (int): 0始まりのフレームインデックス
            figsize (tuple): 図のサイズ (幅, 高さ)
            cmap (str): 'gray' 等のカラーマップ (必要な場合)
        """
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"Frame index {idx} is out of range [0, {len(self.frames)-1}]")

        frame = self.frames[idx]

        # OpenCV で BGR 保存しているなら RGB に変換
        # if frame.ndim == 3 and frame.shape[2] == 3:
        #     frame = frame[..., ::-1]

        plt.figure(figsize=figsize)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Frame {idx} / {len(self.frames)}")
        plt.show(block=True)

npy_path = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/raw_event_data.npy"
trigger_npy_path = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/triggers.npy"
mask_video_path = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/output_video_with_eye_points.mp4"

wave_cluster = waveCluster(npy_path, trigger_npy_path)
wave_cluster.test()



# wave_cluster.load_video_frames(mask_video_path)
# wave_cluster.show_frame(idx=100)