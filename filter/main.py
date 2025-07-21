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

class SNNFilter:
    def __init__(self, input_raw_path):
        self.input_raw_path = input_raw_path
        self.H, self.W = 720, 1280
        # LIF層を定義
        beta = 0.8
        threshold = 1.0
        self.lif = SNN.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
        self.mem = None # 膜電位状態をフレーム間で保持

    def load_evt3(self, time_window=10000):
        """
        EVT3ファイルを読み込み、NumPy配列として返すメソッド
        """
        reader = RawReader(self.input_raw_path)
        self.raw_event_data = []
        
        while not reader.is_done():
            events = reader.load_delta_t(time_window)
            if events is not None:
                self.raw_event_data.append(events)
        
        self.raw_event_data = np.concatenate(self.raw_event_data)

        # ここで (N,) のタプル配列 → (N,4) の int 配列に変換
        self.raw_event_data = np.array(self.raw_event_data.tolist(), dtype=int)
        
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
        for frame_idx in range(total_frames):
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
        dt_us = 1000
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
    
    def filter_3d(self):
        # filtered_events.npyを読み込む
        events = np.load('snn_filtered_events.npy')
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
        
        if filtered_events:
            self.filtered_3d_events = np.concatenate(filtered_events, axis=0)
        else:
            self.filtered_3d_events = np.empty((0, 4))  # 4列の空配列

        print(f"---------------3dフィルタリング終了----------------")
        print("イベントの形状:", self.filtered_3d_events.shape)  # 例: (N, 4)
        print("最初の5つのイベント:\n", self.filtered_3d_events[:5])
        print("最後の5つのイベント:\n", self.filtered_3d_events[-5:])
        np.save("3d_filtered_events.npy", self.filtered_3d_events)
        print(f"3d_filtered_events.npyに保存しました ({len(self.filtered_3d_events)}件)")
        
# イベントデータの取得
# raw_path = "/home/carrobo2024/00_drowsiness_ws/recording_250321_143628_260.raw"
# raw_path = "/home/carrobo2024/00_drowsiness_ws/video/02/recording_250529_212441_423.raw"
raw_path = "/home/carrobo2024/00_drowsiness_ws/video/02/recording_250529_212441_423.raw"
snn_filter = SNNFilter(raw_path)
snn_filter.load_evt3()
snn_filter.events_to_video_frames(snn_filter.raw_event_data, 'raw_event_data.mp4')
snn_filter.snn_filter()
snn_filter.events_to_video_frames(snn_filter.snn_filtered_events, 'snn_filtered_events.mp4')
snn_filter.filter_3d()
snn_filter.events_to_video_frames(snn_filter.filtered_3d_events, '3d_filtered_events.mp4')

