from metavision_core.event_io.raw_reader import RawReader
import numpy as np
import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from celluloid import Camera
from matplotlib.animation import FFMpegWriter
import cv2

class SNNFilter:
    def __init__(self, input_raw_path):
        self.input_raw_path = input_raw_path
        self.H, self.W = 720, 1280
        # 何μs ごとにスパイクを「時間ステップ」化するかを決める
        self.time_bin_us = 1000   # 1ms ごと

    def load_evt3(self, time_window=10000):
        """
        EVT3ファイルを読み込み、NumPy配列として返すメソッド
        """
        reader = RawReader(self.input_raw_path)
        self.event_data = []
        
        while not reader.is_done():
            events = reader.load_delta_t(time_window)
            if events is not None:
                self.event_data.append(events)
        
        self.event_data = np.concatenate(self.event_data)

        # ここで (N,) のタプル配列 → (N,4) の int 配列に変換
        self.event_data = np.array(self.event_data.tolist(), dtype=int)
        
        print(f"Loaded {len(self.event_data)} events.")
        print("First 10 events:")
        print(self.event_data[:10])
        print("Last 10 events:")
        print(self.event_data[-10:])
        # print(f"len: {len(self.event_data)}")
        # print(f"dtype: {self.event_data.dtype}")

        self.trigger_events = []

        try:
            self.trigger_events = reader.get_ext_trigger_events()
            # print(f"trigger events: {self.trigger_events}")
            # print(f"trigger events num: {len(self.trigger_events)}")
        except Exception as e:
            print("トリガーイベント取得中にエラーが発生しました:", e)
        
        return self.event_data
    
    def events_to_video_frames(self, output_path: str='spile.mp4', fps=30, H=720, W=1280):
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
        t_min = int(self.event_data[:,3].min())
        t_max = int(self.event_data[:,3].max())

        # 各イベント時刻を t_min からの相対値に正規化
        norm_times = self.event_data[:,3] - t_min

        # 総フレーム数を算出（端数は切り上げ）
        total_frames = ((t_max - t_min) + frame_duration_us - 1) // frame_duration_us
        
        # OpenCV の VideoWriter を初期化
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=True)
        
        # 各フレームを順に生成・書き込み
        for frame_idx in range(total_frames):
            # このフレームが担当する時間区間 [start_t, end_t)
            start_t = frame_idx * frame_duration_us
            end_t   = start_t + frame_duration_us
            
            # 時間窓内のイベントを抽出
            mask = (norm_times >= start_t) & (norm_times < end_t)
            ev = self.event_data[mask]
            
            # 白背景の RGB 画像を用意
            frame_img = np.full((H, W, 3), 255, dtype=np.uint8)
            
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

    def events_to_spike_tensor(self):
        self.event_data = np.array(self.event_data.tolist(), dtype=int)
        max_t = int(self.event_data[:, 3].max())      # 最も大きいタイムスタンプ
        T = (max_t // self.time_bin_us) + 1       # 0始まりカウント → +1

        # [T, B, C, H, W]、ここではバッチサイズ B=1、チャンネル C=2 (OFF/ON)
        self.spike_tensor = torch.zeros(T, 1, 2, self.H, self.W, dtype=torch.uint8)

        for x, y, p, t in self.event_data.astype(int):
            bin_idx = t // self.time_bin_us
            ch = int(p)            # polarity 0→チャンネル0, 1→チャンネル1
            self.spike_tensor[bin_idx, 0, ch, y, x] = 1
        
        print("spike_tensor:")
        print(self.spike_tensor)
        print("spike_tensor shape:", self.spike_tensor.shape)
    
    def events_to_spike_chunks(self, chunk_size_bins=1000):
        """
        イベントデータをチャンク単位でスパイクテンソル化して返す。
        chunk_size_bins: 1チャンクあたりの時間ビン数 (デフォルト1000＝1s分)
        """
        # (1) タプルリスト→(N,4) 数値配列
        self.event_data = np.array(self.event_data.tolist(), dtype=int)

        # (2) 全ビン数の算出
        max_t = int(self.event_data[:, 3].max())
        T = (max_t // self.time_bin_us) + 1

        # (3) 各イベントのbin_idxを先に計算しておく
        bin_indices = (self.event_data[:, 3] // self.time_bin_us).astype(int)

        # (4) チャンク数
        num_chunks = (T + chunk_size_bins - 1) // chunk_size_bins

        spike_chunks = []
        for ci in range(num_chunks):
            start_bin = ci * chunk_size_bins
            end_bin = min((ci + 1) * chunk_size_bins, T)
            cur_size = end_bin - start_bin

            # [cur_size, B=1, C=2, H, W]
            chunk_tensor = torch.zeros(cur_size, 1, 2, self.H, self.W, dtype=torch.uint8)

            # このチャンクに含まれるイベントのみ処理
            mask = (bin_indices >= start_bin) & (bin_indices < end_bin)
            ev_chunk = self.event_data[mask]
            bins_chunk = bin_indices[mask] - start_bin

            for (x, y, p, _), b in zip(ev_chunk, bins_chunk):
                chunk_tensor[b, 0, p, y, x] = 1

            spike_chunks.append(chunk_tensor)
            print(f"Chunk {ci+1}/{num_chunks}: bins {start_bin}–{end_bin-1}, shape {chunk_tensor.shape}")

        return spike_chunks
    
    def events_to_spike_chunks_with_anim(self,
                                         chunk_size_bins: int = 100,
                                         save_path: str = "spikes.gif",
                                         interval_ms: int = 50):
        """
        1) イベントデータをチャンク単位でスパイクテンソル化
        2) 全チャンクをまとめてアニメーション生成し保存
        """
        # --- (1) タプルリスト→(N,4) 数値配列 ---
        self.event_data = np.array(self.event_data.tolist(), dtype=int)

        # --- (2) 全ビン数とインデックス ---
        max_t      = int(self.event_data[:, 3].max())
        T          = (max_t // self.time_bin_us) + 1
        bin_indices = (self.event_data[:, 3] // self.time_bin_us).astype(int)

        # --- (3) チャンク数と準備 ---
        num_chunks = (T + chunk_size_bins - 1) // chunk_size_bins
        spike_chunks = []

        # Matplotlib + Celluloid の準備
        fig, ax = plt.subplots()
        cam = Camera(fig)

        # --- (4) チャンクごとにテンソル化 & アニメーション用フレーム撮影 ---
        for ci in range(num_chunks):
            start = ci * chunk_size_bins
            end   = min((ci + 1) * chunk_size_bins, T)
            cur_T = end - start

            # チャンク用テンソル
            chunk_tensor = torch.zeros(cur_T, 1, 2, self.H, self.W, dtype=torch.uint8)

            # チャンク内イベントのみ振り分け
            mask     = (bin_indices >= start) & (bin_indices < end)
            ev_chunk = self.event_data[mask]
            bins     = bin_indices[mask] - start

            for (x, y, p, _), b in zip(ev_chunk, bins):
                chunk_tensor[b, 0, p, y, x] = 1

            spike_chunks.append(chunk_tensor)
            print(f"Chunk {ci+1}/{num_chunks}: bins {start}–{end-1}, shape {chunk_tensor.shape}")

            # アニメーション用：ON チャンネルのみ可視化
            # for t in range(cur_T):
            #     ax.clear()
            #     ax.imshow(chunk_tensor[t, 0, 1], cmap="gray", vmin=0, vmax=1)
            #     ax.set_title(f"Chunk {ci+1}/{num_chunks}, Frame {t+1}/{cur_T}")
            #     ax.axis("off")
            #     cam.snap()
            # フレームを間引いてキャプチャ（例：10 分の 1 の頻度）
            for t in range(cur_T):
                if t % 10 != 0:
                    continue
                ax.clear()
                ax.imshow(chunk_tensor[t, 0, 1], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"Chunk {ci+1}/{num_chunks}, Frame {t+1}/{cur_T}")
                ax.axis("off")
                cam.snap()

        # --- (5) アニメーション保存 ---
        anim = cam.animate(interval=interval_ms)
        # anim.save(save_path, writer='pillow')
        writer = FFMpegWriter(fps=1000//interval_ms, metadata=dict(artist='Me'), bitrate=1800)
        anim.save("spikes.mp4", writer=writer)
        print("Saved animation to spikes.mp4")

        return spike_chunks

    def snn_filter(self):
        ev_sorted = self.event_data[np.argsort(self.event_data[:,3])]

        dt_us = 1
        t_min = int(self.event_data[:,3].min())
        t_max = int(self.event_data[:,3].max())
        num_windows = ((t_max - t_min) + dt_us - 1) // dt_us

        self.filtered_events = []

        for w in range(num_windows):
            print(f"{w}/{num_windows}")

            t_start = t_min + w * dt_us
            t_stop  = t_start + dt_us

            # このウィンドウのイベント
            mask = (ev_sorted[:,3] >= t_start) & (ev_sorted[:,3] < t_stop)
            ev_win = ev_sorted[mask]

            # スパイクテンソル化 [1,2,H,W]
            spike = torch.zeros(1, 2, self.H, self.W, dtype=torch.uint8)
            for x, y, p, t in ev_win.astype(int):
                spike[0, p, y, x] = 1

            # SNNフィルタ
            spk_out, self.mem = self.lif(spike, self.mem)  # spk_out: [1,2,H,W]
            spk = spk_out[0].cpu().numpy()  # [2,H,W]

            # (x,y,p,t_us) 形式に戻す
            ys_p, xs_p = np.nonzero(spk[1])
            ys_n, xs_n = np.nonzero(spk[0])
            if xs_p.size:
                ts = np.full(xs_p.shape, t_start, int)
                self.filtered_events.append(np.stack([xs_p, ys_p, np.zeros_like(xs_p), ts], axis=1))
            if xs_n.size:
                ts = np.full(xs_n.shape, t_start, int)
                self.filtered_events.append(np.stack([xs_n, ys_n, np.ones_like(xs_n), ts], axis=1))

        self.filtered_events = np.vstack(self.filtered_events).astype(int)

        print(f"Filtered {len(self.filtered_events)} events.")
        print("First 10 events:")
        print(self.filtered_events[:10])
        print("Last 10 events:")
        print(self.filtered_events[-10:])

    def snn_filter_fast(self, dt_us=1000):
        ev = self.event_data
        t0 = int(ev[:,3].min())
        bin_idx = ((ev[:,3] - t0) // dt_us).astype(int)
        num_w = bin_idx.max() + 1

        # ウィンドウ→イベントインデックス
        ev_per_w = [[] for _ in range(num_w)]
        for i, b in enumerate(bin_idx):
            ev_per_w[b].append(i)

        # LIF を GPU に載せたい場合：
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device.type: {device.type}")
        self.lif.to(device)
        mem = None

        # ベースの空スパイクテンソル
        base = torch.zeros(1,2,self.H,self.W, dtype=torch.uint8, device=device)

        out_events = []
        for w in range(num_w):
            print(f"{w}/{num_w}")
            idxs = ev_per_w[w]
            if not idxs:
                # イベントなしウィンドウでも SNN をステップ更新
                spike = base.clone()
            else:
                arr = ev[idxs].astype(int)
                xs, ys, ps = arr[:,0], arr[:,1], arr[:,2]
                spike = base.clone()
                spike[0, ps, ys, xs] = 1

            # SNNフィルタ（GPU or CPU）
            if device.type=="cuda":
                spike = spike.to(device)
            spk_out, mem = self.lif(spike, mem)
            spk = spk_out[0].cpu().numpy()

            # イベント形式に復元
            ys_p, xs_p = np.nonzero(spk[1])
            ys_n, xs_n = np.nonzero(spk[0])
            if xs_p.size:
                ts = np.full(xs_p.shape, t0 + w*dt_us, int)
                out_events.append(np.stack([xs_p, ys_p, np.zeros_like(xs_p), ts],1))
            if xs_n.size:
                ts = np.full(xs_n.shape, t0 + w*dt_us, int)
                out_events.append(np.stack([xs_n, ys_n, np.ones_like(xs_n), ts],1))

        self.filtered_events = np.vstack(out_events).astype(int)

        print(f"Filtered {len(self.filtered_events)} events.")
        print("First 10 events:")
        print(self.filtered_events[:10])
        print("Last 10 events:")
        print(self.filtered_events[-10:])

        np.save("filtered_events.npy", self.filtered_events)

    def events_to_filtered_video(self, output_path: str='filtered.mp4', fps=30):
        """
        各フレームごとに SNN フィルタを適用し、動画として保存します。
        """
        # 1フレーム時間幅(μs)
        frame_duration_us = int(1e6 / fps)
        t_min = int(self.event_data[:,3].min())
        t_max = int(self.event_data[:,3].max())
        norm_times = self.event_data[:,3] - t_min
        total_frames = ((t_max - t_min) + frame_duration_us - 1) // frame_duration_us

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.W, self.H), isColor=True)

        for idx in range(total_frames):
            start_t = idx * frame_duration_us
            end_t = start_t + frame_duration_us
            mask = (norm_times >= start_t) & (norm_times < end_t)
            ev = self.event_data[mask]

            # スパイクテンソル[1,1,2,H,W]を作成
            spike = torch.zeros(1, 1, 2, self.H, self.W, dtype=torch.uint8)
            if ev.size > 0:
                # 同一ピクセル最後優先で1に設定
                ev_sorted = ev[np.argsort(ev[:,3])]
                for x, y, p, _ in ev_sorted:
                    spike[0, 0, p, y, x] = 1

            # SNNフィルタ適用
            spk_out, self.mem = self.lif(spike[0], self.mem)
            spk_np = spk_out.cpu().numpy()[0]  # [2, H, W]

            # RGBフレーム生成
            frame_img = np.full((self.H, self.W, 3), 0, dtype=np.uint8)
            # positive→red, negative→blue
            frame_img[spk_np[1] == 1] = (0, 0, 255)
            frame_img[spk_np[0] == 1] = (255, 0, 0)

            out.write(frame_img)

        out.release()
        print(f"{output_path} に保存しました （{total_frames}フレーム、{fps}fps）")

    def snn_filter(self):
        ev_sorted = self.event_data[np.argsort(self.event_data[:,3])]

        dt_us = 1000
        t_min = int(self.event_data[:,3].min())
        t_max = int(self.event_data[:,3].max())
        num_windows = ((t_max - t_min) + dt_us - 1) // dt_us

        self.filtered_events = []

        for w in range(num_windows):
            print(f"{w}/{num_windows}")

            t_start = t_min + w * dt_us
            t_stop  = t_start + dt_us

            # このウィンドウのイベント
            mask = (ev_sorted[:,3] >= t_start) & (ev_sorted[:,3] < t_stop)
            ev_win = ev_sorted[mask]

            # スパイクテンソル化 [1,2,H,W]
            spike = torch.zeros(1, 2, self.H, self.W, dtype=torch.uint8)
            for x, y, p, t in ev_win.astype(int):
                spike[0, p, y, x] = 1

            # SNNフィルタ
            spk_out, self.mem = self.lif(spike, self.mem)  # spk_out: [1,2,H,W]
            spk = spk_out[0].cpu().numpy()  # [2,H,W]

            # (x,y,p,t_us) 形式に戻す
            ys_p, xs_p = np.nonzero(spk[1])
            ys_n, xs_n = np.nonzero(spk[0])
            if xs_p.size:
                ts = np.full(xs_p.shape, t_start, int)
                self.filtered_events.append(np.stack([xs_p, ys_p, np.zeros_like(xs_p), ts], axis=1))
            if xs_n.size:
                ts = np.full(xs_n.shape, t_start, int)
                self.filtered_events.append(np.stack([xs_n, ys_n, np.ones_like(xs_n), ts], axis=1))

        self.filtered_events = np.vstack(self.filtered_events).astype(int)

        np.save("filtered_events.npy", self.filtered_events)




# イベントデータの取得
file_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260.raw"
snn_filter = SNNFilter(file_path)
snn_filter.load_evt3()
snn_filter.events_to_video_frames()
# snn_filter.events_to_spike_tensor()
# snn_filter.events_to_spike_chunks()
# chunks = snn_filter.events_to_spike_chunks_with_anim()

# データの整形
# .aedat, .mat, .dat など　▶　


# SNNの定義
import snntorch as SNN
from snntorch import surrogate

# 1. モデル定義：単一の LIF ニューロン層
beta = 0.8          # 膜電位の減衰率 (論文の 1 - λ 相当)
threshold = 1.0     # 閾値 θ
lif = SNN.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())


# SNNの実行


# フィルタをかけたデータを保存


