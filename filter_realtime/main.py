from metavision_core.event_io.raw_reader import RawReader
import numpy as np
import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from celluloid import Camera
from matplotlib.animation import FFMpegWriter
import cv2
import snntorch as SNN
from snntorch import surrogate
from tqdm import tqdm
import glob
import time

class SNNFilter:
    def __init__(self, input_raw_path):
        self.input_raw_path = input_raw_path
        self.H, self.W = 720, 1280
        # LIF層を定義
        beta = 0.8 # 過去の膜電位をどれだけ保持するか
        threshold = 1.0 # スパイクを出力するための閾値
        self.lif = SNN.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
        self.mem = None # 膜電位状態をフレーム間で保持

    def load_evt3(self, save_npy_path, time_window=10000):
        """
        EVT3ファイルを読み込み、NumPy配列として返すメソッド
        """
        reader = RawReader(self.input_raw_path)
        self.raw_event_data = []
        
        pbar = tqdm(desc="Loading EVT3 chunks", unit="chunk")
        while not reader.is_done():
            events = reader.load_delta_t(time_window)
            if events is not None:
                self.raw_event_data.append(events)
            pbar.update(1)

        pbar.close()
        self.raw_event_data = np.concatenate(self.raw_event_data)

        # ここで (N,) のタプル配列 → (N,4) の int 配列に変換
        self.raw_event_data = np.array(self.raw_event_data.tolist(), dtype=int)
        
        np.save(save_npy_path, self.raw_event_data)

        print(f"Loaded {len(self.raw_event_data)} events.")
        print("First 10 events:")
        print(self.raw_event_data[:10])
        print("Last 10 events:")
        print(self.raw_event_data[-10:])
        # print(f"len: {len(self.event_data)}")
        # print(f"dtype: {self.event_data.dtype}")
        
        return self.raw_event_data
    
    def events_to_video_frames(self, event_data, output_path, fps=30):
        """
        イベントデータから動画フレームを生成して MP4 に保存します。
        
        パラメータ：
            fps (int): 動画のフレームレート（1秒あたりのフレーム数）
            H, W (int): センサの画像サイズ（高さ H, 幅 W）
        
        処理の流れ：
          1. fps から 1フレームあたりの時間幅（μs）を計算
          2. イベント時刻を最小時刻で正規化し、総フレーム数を算出
          3. OpenCV の VideoWriter を初期化
          4. 各フレームごとに：
             a. その時間窓に含まれるイベントを抽出
             b. 白背景の画像を生成
             c. 同一ピクセルで後出しのイベントを優先して、
                プラス(0)は赤、マイナス(1)は青で塗りつぶし
             d. フレームを動画に書き込み
          5. 動画ファイルをクローズして保存完了
        """
        # 1フレームの時間幅をマイクロ秒単位で計算
        frame_duration_us = int(1e6 / fps)
        
        # イベントの最小／最大時刻を取得
        t_min = int(event_data[:,3].min())
        t_max = int(event_data[:,3].max())

        # 各イベント時刻を t_min からの相対値に正規化
        norm_times = event_data[:,3] - t_min

        # 総フレーム数を算出（端数は切り上げ）
        total_frames = ((t_max - t_min) + frame_duration_us - 1) // frame_duration_us
        
        # OpenCV の VideoWriter を初期化
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.W, self.H), isColor=True)
        
        # 各フレームを順に生成・書き込み
        # for frame_idx in range(total_frames):
        for frame_idx in tqdm(range(total_frames), desc="Rendering raw→MP4", unit="frame"):
            # このフレームが担当する時間区間 [start_t, end_t)
            start_t = frame_idx * frame_duration_us
            end_t   = start_t + frame_duration_us
            
            # 時間窓内のイベントを抽出
            mask = (norm_times >= start_t) & (norm_times < end_t)
            ev = event_data[mask]
            
            # 白背景の RGB 画像を用意
            frame_img = np.full((self.H, self.W, 3), 0, dtype=np.uint8)
            
            if ev.size > 0:
                # 時系列順にソートし、同一ピクセルは後のイベントで上書き
                ev_sorted = ev[np.argsort(ev[:,3])]
                for x, y, p, t in ev_sorted:
                    if p == 0:
                        # プラス極性：赤 (BGR=(0,0,255))
                        frame_img[y, x] = (0, 0, 255)
                    else:
                        # マイナス極性：青 (BGR=(255,0,0))
                        frame_img[y, x] = (255, 0, 0)
            
            # フレームを動画に書き込む
            out.write(frame_img)
        
        # 動画ファイルをクローズ
        out.release()
        print(f"{output_path} に保存しました （{total_frames} フレーム、{fps} fps）")

    def snn_filter(self):
        ev_sorted = self.raw_event_data[np.argsort(self.raw_event_data[:,3])]
        dt_us = 1000 # SNNフィルタ処理の時間窓
        t_min = int(self.raw_event_data[:,3].min())
        t_max = int(self.raw_event_data[:,3].max())
        num_windows = ((t_max - t_min) + dt_us - 1) // dt_us

        # 事前にグループごとのインデックスリストを作成
        bin_idx = ((ev_sorted[:,3] - t_min) // dt_us).astype(int)
        group_indices = [np.where(bin_idx == w)[0] for w in range(num_windows)]

        # デバイス設定と LIF レイヤ準備
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lif.to(device)
        self.mem = None

        # ベーススパイクテンソルを一度だけ生成
        base_spike = torch.zeros(1, 2, self.H, self.W, dtype=torch.uint8, device=device)

        filtered_events = []
        for w in range(num_windows):
            print(f"{w}/{num_windows}")
            t_start = t_min + w * dt_us
            t_stop  = t_start + dt_us

            # テンソルをゼロクリア
            spike = base_spike.zero_()
            idxs = group_indices[w]
            if idxs.size:
                grp = ev_sorted[idxs].astype(int)
                spike[0, grp[:,2], grp[:,1], grp[:,0]] = 1

            # SNNフィルタ呼び出し
            spike = spike.to(device)
            spk_out, self.mem = self.lif(spike, self.mem)
            spk = spk_out[0].cpu().numpy()  # [2,H,W]

            # (x,y,p,t_us) 形式に戻す
            ys_p, xs_p = np.nonzero(spk[1])
            ys_n, xs_n = np.nonzero(spk[0])
            if xs_p.size:
                ts = np.full(xs_p.shape, t_start, int)
                filtered_events.append(np.stack([xs_p, ys_p, np.zeros_like(xs_p), ts], axis=1))
            if xs_n.size:
                ts = np.full(xs_n.shape, t_start, int)
                filtered_events.append(np.stack([xs_n, ys_n, np.ones_like(xs_n), ts], axis=1))

        self.snn_filtered_events = np.concatenate(filtered_events, axis=0)
        np.save("snn_filtered_events.npy", self.snn_filtered_events)
        print(f"snn_filtered_events.npyに保存しました ({len(self.snn_filtered_events)}件)")

    def snn_filter_sequential(self, output_path, dt_us=1000):
        """
        EVT3ファイルを逐次的に読み込みながら、SNNでフィルタ処理を行う。
        イベント全体をメモリに保持せず、メモリ効率を大幅に改善する。
        """
        start_time = time.perf_counter()

        reader = RawReader(self.input_raw_path)

        t_min = int(self.raw_event_data[:,3].min())
        t_max = int(self.raw_event_data[:,3].max())
        total_windows = ((t_max - t_min) + dt_us - 1) // dt_us

        # SNNの初期化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lif.to(device)
        self.mem = None

        base_spike = torch.zeros(1, 2, self.H, self.W, dtype=torch.uint8, device=device)
        filtered_events = []
        chunk_id = 0

        t_global = None  # 絶対時刻（出力イベントの時刻に使用）

        CHUNK_SIZE = 1000  # 何ウィンドウごとに一時保存するか

        frame_idx = 0
        pbar = tqdm(total=total_windows, desc="SNN sequential filtering", unit="window")
        while not reader.is_done():
            events = reader.load_delta_t(dt_us)
            if events is None or len(events) == 0:
                frame_idx += 1
                pbar.update(1)
                continue

            ev_np = np.array(events.tolist(), dtype=int)
            if ev_np.size == 0:
                continue

            # 開始時刻の記録（初回のみ）
            if t_global is None:
                t_global = ev_np[0, 3]

            # spikeテンソル初期化
            spike = base_spike.zero_()
            spike[0, ev_np[:,2], ev_np[:,1], ev_np[:,0]] = 1

            # SNNフィルタ処理
            with torch.no_grad():
                spk_out, mem_next = self.lif(spike, self.mem)
            self.mem = mem_next.detach()
            del mem_next
            # spk_out, self.mem = self.lif(spike, self.mem)
            # spk = spk_out[0].cpu().numpy()  # shape: [2, H, W]
            spk = spk_out[0].detach().cpu().numpy()
            del spk_out

            # 出力イベント生成 (x, y, p, t)
            ys_p, xs_p = np.nonzero(spk[1])  # polarity=0 (positive)
            ys_n, xs_n = np.nonzero(spk[0])  # polarity=1 (negative)

            t_us = t_global + frame_idx * dt_us
            if xs_p.size:
                ts = np.full(xs_p.shape, t_us, int)
                filtered_events.append(np.stack([xs_p, ys_p, np.zeros_like(xs_p), ts], axis=1))
            if xs_n.size:
                ts = np.full(xs_n.shape, t_us, int)
                filtered_events.append(np.stack([xs_n, ys_n, np.ones_like(xs_n), ts], axis=1))

            frame_idx += 1

            if frame_idx % CHUNK_SIZE == 0:
                chunk_array = np.concatenate(filtered_events, axis=0)
                np.save(f"snn_chunk_{chunk_id:03d}.npy", chunk_array)
                filtered_events.clear()
                chunk_id += 1

            # print(f"Processed frame {frame_idx} ({len(ev_np)} events in, {xs_p.size + xs_n.size} events out)")
            pbar.update(1)
            pbar.set_postfix({"in":len(ev_np),"out":xs_p.size+xs_n.size})
        pbar.close()

        all_chunks = sorted(glob.glob("snn_chunk_*.npy"))
        arrays = [np.load(f) for f in all_chunks]
        self.snn_filtered_events = np.concatenate(arrays, axis=0)
        np.save(output_path, self.snn_filtered_events)
        print(f"ffffffffff")

        # チャンクごとにnpyを保存するため一括用のプログラムはコメントアウト
        # 保存処理
        # if filtered_events:
        #     self.snn_filtered_events = np.concatenate(filtered_events, axis=0)
        # else:
        #     self.snn_filtered_events = np.empty((0, 4), dtype=int)

        # np.save(output_path, self.snn_filtered_events)
        # print(f"{output_path} に保存しました ({len(self.snn_filtered_events)}件)")
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"snnフィルタにかかった時間: {elapsed:.2f} 秒")

    def filter_3d(self, load_npy_path, save_npy_path):
        start_time = time.perf_counter()
        # filtered_events.npyを読み込む
        events = np.load(load_npy_path)
        print(f"snnでフィルタリングされたイベントデータを取得しました")
        print("イベントの形状:", events.shape)  # 例: (N, 4)
        print("最初の5つのイベント:\n", events[:5])
        print("最後の5つのイベント:\n", events[-5:])

        t_min = events[:, 3].min()
        t_max = events[:, 3].max()
        current_start = t_min

        buffer_duration = 10000 # us = 10ms
        threshold = 100
        filtered_events = []
        step = 1000 # us = 1 ms

        total_steps = ((events[:,3].max() - events[:,3].min()) // step) + 1
        pbar = tqdm(total=total_steps, desc="3Dフィルタ", unit="step")

        while current_start + buffer_duration <= t_max:
            current_end = current_start + buffer_duration

            # 10ms分のイベントを取得（スライディングバッファ）
            buffer_events = events[(events[:, 3] >= current_start) & 
                                        (events[:, 3] < current_end)]

            # イベント数を数える
            s = len(buffer_events)

            # イベント数がしきい値を超えた場合のみ保存
            if s > threshold:
                # print(f"s:{s} t:{current_start}-{current_end}")
                one_ms_events = events[
                    (events[:, 3] >= current_start) &
                    (events[:, 3] < current_start + step)]
                filtered_events.append(one_ms_events)

            # スライディングバッファを更新（1msスライド）
            current_start += step

            pbar.update(1)
        
        pbar.close()
        
        if filtered_events:
            self.filtered_3d_events = np.concatenate(filtered_events, axis=0)
        else:
            self.filtered_3d_events = np.empty((0, 4))  # 4列の空配列

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"3dフィルタにかかった時間: {elapsed:.2f} 秒")

        print(f"---------------3dフィルタリング終了----------------")
        print("イベントの形状:", self.filtered_3d_events.shape)  # 例: (N, 4)
        print("最初の5つのイベント:\n", self.filtered_3d_events[:5])
        print("最後の5つのイベント:\n", self.filtered_3d_events[-5:])
        np.save(save_npy_path, self.filtered_3d_events)
        print(f"{save_npy_path}に保存しました ({len(self.filtered_3d_events)}件)")
        
# イベントデータの取得
# raw_path = "/home/carrobo2024/00_drowsiness_ws/recording_250321_143628_260.raw"
# raw_path = "/home/carrobo2024/00_drowsiness_ws/video/02/recording_250529_212441_423.raw"
raw_path = "/home/carrobo2024/00_drowsiness_ws/video/02/recording_250529_212441_423.raw"
# raw_path = "/home/carrobo2024/00_drowsiness_ws/video/dynamic_objects_in_the_background/recording_250708_223130_063.raw"
snn_filter = SNNFilter(raw_path)


# save_npy_path="/home/carrobo2024/00_drowsiness_ws/video/dynamic_objects_in_the_background/raw_event_data.npy"
save_npy_path="/home/carrobo2024/00_drowsiness_ws/video/02/raw_event_data.npy"
snn_filter.load_evt3(save_npy_path)


output_path = "/home/carrobo2024/00_drowsiness_ws/video/02/raw_event_data.mp4"
# snn_filter.events_to_video_frames(snn_filter.raw_event_data, output_path)

# output_path="/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/snn_filtered_events.npy"
output_path="/home/carrobo2024/00_drowsiness_ws/video/02/snn_filtered_events.npy"
snn_filter.snn_filter_sequential(output_path)

# output_path = "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/snn_filtered_events.mp4"
output_path = "/home/carrobo2024/00_drowsiness_ws/video/02/snn_filtered_events.mp4"
snn_filter.events_to_video_frames(snn_filter.snn_filtered_events, output_path)

# load_npy_path="/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/snn_filtered_events.npy"
# save_npy_path="/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/3d_filtered_events.npy"
load_npy_path="/home/carrobo2024/00_drowsiness_ws/video/02/snn_filtered_events.npy"
save_npy_path="/home/carrobo2024/00_drowsiness_ws/video/02/3d_filtered_events.npy"
snn_filter.filter_3d(
    load_npy_path,
    save_npy_path)

# output_path = '/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/3d_filtered_events.mp4'
output_path = '/home/carrobo2024/00_drowsiness_ws/video/02/3d_filtered_events.mp4'
snn_filter.events_to_video_frames(snn_filter.filtered_3d_events, output_path)

