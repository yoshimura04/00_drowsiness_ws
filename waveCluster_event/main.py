from metavision_core.event_io.raw_reader import RawReader
import numpy as np
import matplotlib.pyplot as plt
import pywt
from collections import deque
from scipy.ndimage import label

class waveCluster:
    def __init__(self, input_raw_path):
        self.input_raw_path = input_raw_path
        self.H, self.W = 720, 1280
    
    def test(self,
            buffer_time=100000,
            step_time=100000,
            grid_H=360,
            grid_W=640,):

        # グリッドセルあたりのピクセル幅・高さ
        cell_h = self.H / grid_H
        cell_w = self.W / grid_W
        n_cells = grid_H * grid_W
        
        reader = RawReader(self.input_raw_path)

        # ----------初回のウィンドウを取得------------

        events = reader.load_delta_t(buffer_time)
        events = np.array(events.tolist(), dtype=int)
        print(events)

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

            cnt_flat = np.bincount(flat, minlength=n_cells)
            count_mat = cnt_flat.reshape((grid_H, grid_W))

        # -----------ウィンドウを1msずつスライド--------------

        current_start = 0

        while True:
            next_start = current_start + step_time

            # eventsの古い1ms分を消す
            while buffer_deque and buffer_deque[0][0] < next_start:
                tsi_old, idx_old = buffer_deque.popleft()
                cnt_flat[idx_old] -= 1

            # eventsに新しい1ms分を足す
            events = reader.load_delta_t(step_time)
            events = np.array(events.tolist(), dtype=int)
            print(events)

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
            self.plot_count_mat(count_mat)

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
            plt.figure(figsize=(6,4))
            plt.imshow(cA1, cmap='jet')
            plt.title(f"Wavelet Approximation (level1) @ t={current_start}")
            plt.colorbar(label="Coefficient")
            plt.tight_layout()
            plt.show()

            # -------------しきい値でバイナリマスクを作成---------------

            # 単純に平均＋α×標準偏差で閾値を定義する例
            # mean_A = np.mean(cA1)
            # std_A  = np.std(cA1)
            # alpha = 1.0  # 調整パラメータ
            # threshold = mean_A + alpha * std_A
            threshold = 30

            # 2) バイナリマスクを作成（True がクラスタ候補領域）
            mask = cA1 > threshold

            plt.figure(figsize=(6,4))
            plt.imshow(mask, cmap='gray')
            plt.title(f"Binary Mask @ t={current_start}")
            plt.tight_layout()
            plt.show()

            # -------------6近傍でラベリング（step5）----------------

            structure = np.array([[0,1,0],
                                [1,1,1],
                                [0,1,0]], dtype=bool)
            labeled, num_clusters = label(mask, structure=structure)

            print(f"検出クラスタ数: {num_clusters}")


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
            plt.figure(figsize=(6,4))
            plt.imshow(labeled, cmap='tab20')
            # 各重心に赤い丸を描画
            for (cy, cx) in centroids:
                plt.scatter(cx, cy, s=10, c='black', marker='o', edgecolors='white')
            plt.title(f"Clusters & Centroids @ t={current_start}  (N={num_clusters})")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

            # -------------目と口の判定（従来手法）し描写-------------------

            plt.figure(figsize=(6,4))
            plt.imshow(labeled, cmap='tab20')

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
            plt.tight_layout()
            plt.show()


            current_start = next_start

            # 終了条件
            if reader.is_done():
                break

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
        plt.show()

raw_path = "/home/carrobo2024/00_drowsiness_ws/video/02/recording_250529_212441_423.raw"
wave_cluster = waveCluster(raw_path)
wave_cluster.test()