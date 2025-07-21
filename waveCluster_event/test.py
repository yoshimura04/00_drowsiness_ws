from metavision_core.event_io.raw_reader import RawReader
import numpy as np
import matplotlib.pyplot as plt
import pywt
from collections import deque

class waveCluster:
    def __init__(self, input_raw_path):
        self.input_raw_path = input_raw_path
        self.H, self.W = 720, 1280

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

    def sliding_buffer_count_matrices(self,
                                                buffer_time=10000,
                                                step_time=1000,
                                                grid_H=360,
                                                grid_W=640,
                                                skip_empty=True):
        """
        インクリメンタル更新版: 差分でカウント行列を更新
        結果は self.buffer_count_matrices に格納
        """
        if self.raw_event_data is None:
            raise RuntimeError("まず load_evt3 を呼んでください。")
        ev = self.raw_event_data
        xs = ev[:,0]; ys = ev[:,1]; ts = ev[:,3]
        # grid_H, grid_W
        if grid_H is None: grid_H = self.H
        if grid_W is None: grid_W = self.W
        cell_h = self.H / grid_H
        cell_w = self.W / grid_W
        # 事前にセルインデックス計算
        ix_all = np.clip((xs / cell_w).astype(int), 0, grid_W-1)
        iy_all = np.clip((ys / cell_h).astype(int), 0, grid_H-1)
        flat_idx_all = iy_all * grid_W + ix_all
        timestamps = ts  # ソート済みであること前提
        # 準備
        t_min = timestamps.min()
        t_max = timestamps.max()
        self.buffer_count_matrices = []
        # 初期ウィンドウ
        t0 = t_min
        end0 = t0 + buffer_time
        end_idx = np.searchsorted(timestamps, end0, side='left')
        n_cells = grid_H * grid_W
        # 最初のカウント
        if end_idx > 0:
            cnt_flat = np.bincount(flat_idx_all[:end_idx], minlength=n_cells)
        else:
            cnt_flat = np.zeros(n_cells, dtype=np.int32)
        count_mat = cnt_flat.reshape((grid_H, grid_W))
        if not(skip_empty and end_idx==0):
            self.buffer_count_matrices.append((t0, count_mat.copy()))
        start_idx = 0
        # スライドループ
        while True:
            t0_new = t0 + step_time
            if t0_new + buffer_time > t_max:
                break
            # 古いイベントのインデックス範囲
            start_idx_new = np.searchsorted(timestamps, t0_new, side='left')
            # 新しいイベントのインデックス範囲
            end_time_new = t0_new + buffer_time
            end_idx_new = np.searchsorted(timestamps, end_time_new, side='left')
            # 差分減算: [start_idx:start_idx_new)
            if start_idx_new > start_idx:
                idxs = flat_idx_all[start_idx:start_idx_new]
                # bincountで減算分
                sub_cnt = np.bincount(idxs, minlength=n_cells)
                cnt_flat = cnt_flat - sub_cnt
            # 差分加算: [end_idx:end_idx_new)
            if end_idx_new > end_idx:
                idxs = flat_idx_all[end_idx:end_idx_new]
                add_cnt = np.bincount(idxs, minlength=n_cells)
                cnt_flat = cnt_flat + add_cnt
            # 更新
            count_mat = cnt_flat.reshape((grid_H, grid_W))
            if not(skip_empty and np.all(count_mat==0)):
                self.buffer_count_matrices.append((t0_new, count_mat.copy()))
            # 次へ
            t0, start_idx, end_idx = t0_new, start_idx_new, end_idx_new
        return self.buffer_count_matrices

    def show_buffer_shapes(self, max_display=10):
        """
        self.buffer_count_matrices の概要を表示。
        max_display: 最初の max_display 件まで表示。それ以上ある場合は省略表示。
        """
        bc = self.buffer_count_matrices
        n = len(bc)
        print(f"Total windows: {n}")
        for i, (t0, mat) in enumerate(bc):
            if i >= max_display:
                print(f"... and {n - max_display} more windows")
                break
            print(f"Window {i}: start_time={t0}, matrix shape={mat.shape}")
  
    def visualize_count_matrices(self, max_display=5, cmap='jet'):
        """
        buffer_count_matrices: list of (t0, count_matrix) タプル
        max_display: 可視化する最初のフレーム数
        cmap: カラーマップ名
        """
        for i, (t0, mat) in enumerate(self.buffer_count_matrices[:max_display]):
            plt.figure(figsize=(6, 4))
            plt.imshow(mat, cmap=cmap)
            plt.title(f"Window {i}, start_time={t0}")
            plt.colorbar(label='Event count')
            plt.tight_layout()
            plt.show()
        if len(self.buffer_count_matrices) > max_display:
            print(f"... and {len(self.buffer_count_matrices) - max_display} more matrices not shown")

    def sum_and_visualize_interval(self, start_time=None, duration=1_000_000, cmap='jet'):
        if not hasattr(self, 'buffer_count_matrices') or not self.buffer_count_matrices:
            print("buffer_count_matrices が空です。")
            return
        t0_list = [t0 for (t0, _) in self.buffer_count_matrices]
        t_min = min(t0_list)
        start = t_min if start_time is None else start_time
        end = start + duration
        # 逐次加算
        # grid_H, grid_W は行列サイズに合わせる
        example_mat = self.buffer_count_matrices[0][1]
        grid_H, grid_W = example_mat.shape
        sum_mat = np.zeros((grid_H, grid_W), dtype=np.int32)
        count = 0
        for t0, mat in self.buffer_count_matrices:
            if t0 < start: continue
            if t0 >= end: break
            sum_mat += mat
            count += 1
        if count == 0:
            print(f"{start}～{end} の範囲に行列がありません。")
            return
        plt.figure(figsize=(6,4))
        vmax = np.percentile(sum_mat, 99)
        plt.imshow(sum_mat, cmap=cmap, vmin=0, vmax=vmax)
        plt.title(f"Summed Event Count from {start} to {end} ({count} windows)")
        plt.colorbar(label='Summed event count')
        plt.tight_layout()
        plt.show()
        return sum_mat

    def apply_wavelet_transform(self, wavelet='db1', level=None, mode='symmetric'):
        """
        self.buffer_count_matrices に格納された各 (t0, count_matrix) に対して
        2次元離散ウェーブレット変換を行い、ウェーブレット係数を
        self.wavelet_coeffs に格納する。

        wavelet: PyWavelets で使用するウェーブレット名（例 'db1', 'haar', 'coif1' など）
        level: 分解レベル。None の場合は可能な最大レベルを自動選択。
        mode: 拡張モード（境界処理方法）。PyWavelets の既定は 'symmetric'。
        戻り値として各ウィンドウ開始時刻と対応する係数を返す。
        """
        if not hasattr(self, 'buffer_count_matrices') or not self.buffer_count_matrices:
            raise RuntimeError("まず sliding_buffer_count_matrices() を呼んでバッファ行列を生成してください。")

        coeffs_list = []
        # 代表的な行列サイズで最大分解レベルを計算するために、最初の行列を利用
        # ただし全行列で同じサイズである前提とする
        _, first_mat = self.buffer_count_matrices[0]
        rows, cols = first_mat.shape

        # 使用するウェーブレットオブジェクトを取得
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
        except Exception as e:
            raise ValueError(f"指定されたウェーブレット名が不正です: {wavelet}") from e

        # 自動で最大レベルを決める場合
        if level is None:
            max_level_rows = pywt.dwt_max_level(rows, wavelet_obj.dec_len)
            max_level_cols = pywt.dwt_max_level(cols, wavelet_obj.dec_len)
            # 2次元の場合は行と列の両方で分解可能なレベルの最小値を取る
            max_level = min(max_level_rows, max_level_cols)
        else:
            max_level = level

        for t0, mat in self.buffer_count_matrices:
            # 各 mat が 2D の整数行列である前提
            arr = np.asarray(mat, dtype=float)
            # wavedec2 を使ってレベル max_level で分解
            coeffs = pywt.wavedec2(arr, wavelet=wavelet_obj, mode=mode, level=max_level)
            # coeffs は [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)] のリスト形式
            coeffs_list.append((t0, coeffs))

        # 結果をインスタンス変数に保存
        self.wavelet_coeffs = coeffs_list
        return coeffs_list

    def process_streaming(self,
                          buffer_time=10000,
                          step_time=1000,
                          grid_H=360,
                          grid_W=640,
                          wavelet='db1',
                          level=None,
                          mode='symmetric',
                          save_summary_func=None):
        """
        イベントストリームをチャンク単位で読み込みつつ、
        スライディングウィンドウ（幅buffer_time, ステップ幅step_time）で
        カウント行列を差分更新し、その都度離散ウェーブレット変換を適用する。

        buffer_time: ウィンドウ幅（RawReader のタイムスタンプ単位、通常 µs）
        step_time: ウィンドウをずらす刻み（同上）
        grid_H, grid_W: カウント行列のグリッド解像度
        wavelet: PyWavelets のウェーブレット名
        level: 分解レベル。Noneなら自動計算
        mode: 境界処理モード
        save_summary_func: 各ウィンドウで得られた wavelet 係数を要約保存するコールバック
            形式: save_summary_func(window_start_time, coeffs)
            coeffs は pywt.wavedec2 の返り値
        """

        # グリッドセルあたりのピクセル幅・高さ
        if grid_H is None:
            grid_H = self.H
        if grid_W is None:
            grid_W = self.W
        cell_h = self.H / grid_H
        cell_w = self.W / grid_W
        n_cells = grid_H * grid_W

        # ウェーブレットオブジェクト作成
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
        except Exception as e:
            raise ValueError(f"不正なウェーブレット名です: {wavelet}") from e

        # バッファとして保持する deque: 要素は (ts, flat_idx)
        buffer_deque = deque()
        # カウントベクトル（flat index ごとのイベント数）
        cnt_flat = np.zeros(n_cells, dtype=np.int32)

        # 初期ウィンドウ開始時刻 t0 を決めるため、最初のチャンクを読み込み、buffer_time 以上の範囲を得る
        # ここではまず step_time 刻みでチャンク読み込みを続け、buffer_deque に足していき、
        # 最低でも window_end - window_start >= buffer_time となる状態を作る
        first_window_ready = False
        t0 = None  # ウィンドウ開始時刻
        last_ts = None  # バッファ内最新のタイムスタンプ


        # for in :
            # rawデータから10ms分のイベントデータを取得(スライディングウィンドウを1msずらす)

            # グリッドの準備

            # スライディングウィンドウをグリッドに分け、イベント数をカウント

            # 離散ウェーブレット変換を適用

    def test(self,
            buffer_time=10000,
            step_time=1000,
            grid_H=360,
            grid_W=640,):
    
        # ev = self.raw_event_data
        # # タイムスタンプと座標を取り出す
        # xs = ev[:, 0]
        # ys = ev[:, 1]
        # ts = ev[:, 3]
        # # タイムスタンプでソート
        # order = np.argsort(ts)
        # ts = ts[order]
        # xs = xs[order]
        # ys = ys[order]

        reader = RawReader(self.input_raw_path)
        events = reader.load_delta_t(buffer_time)
        events = np.array(events.tolist(), dtype=int)
        print(events)

        # ----------------初回のウィンドウを取得-------------------

        # start_ts = int(ts[0])
        # end_ts = start_ts + buffer_time

        # # 終了インデックスを取得
        # end_idx = np.searchsorted(ts, end_ts, side='left')
        # print(end_idx)

        # # カウント行列を作成するには、まず抽出した範囲の xs, ys を使ってセルインデックスに変換
        # xs_win = xs[:end_idx]
        # ys_win = ys[:end_idx]

        # # グリッドセルあたりのピクセル幅・高さ
        # cell_h = self.H / grid_H
        # cell_w = self.W / grid_W

        # # 各イベントの列インデックスと行インデックスを計算し、範囲外はクリップ
        # ix = np.clip((xs_win / cell_w).astype(int), 0, grid_W - 1)
        # iy = np.clip((ys_win / cell_h).astype(int), 0, grid_H - 1)

        # # flat index にまとめる
        # flat = iy * grid_W + ix
        # print(flat)

        # # bincount でセルごとのイベント数を得て、2次元に reshape
        # n_cells = grid_H * grid_W
        # cnt_flat = np.bincount(flat, minlength=n_cells)
        # count_mat = cnt_flat.reshape((grid_H, grid_W))

        # # ---------------ウィンドウを1msずつスライド------------------

        # current_start = start_ts
        # # 最後に開始可能なウィンドウは ts[-1] - buffer_time
        # last_start = ts[-1] - buffer_time

        # start_idx = 0

        # while True:
        #     next_start = current_start + step_time
        #     # 次のウィンドウが範囲内か
        #     if next_start > last_start:
        #         break

        #     # 新旧のインデックスを searchsorted で取得
        #     new_start_idx = np.searchsorted(ts, next_start, side='left')
        #     new_end_idx = np.searchsorted(ts, next_start + buffer_time, side='left')

        #     # 古い1ms分（[start_idx:new_start_idx)）を差分減算
        #     if new_start_idx > start_idx:
        #         to_sub = flat[start_idx:new_start_idx]
        #         # bincount で減算分を得る
        #         sub_cnt = np.bincount(to_sub, minlength=n_cells)
        #         cnt_flat = cnt_flat - sub_cnt

    def test2(self,
            buffer_time=10000,
            step_time=1000,
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

        xs = events[:, 0]
        ys = events[:, 1]
        ts = events[:, 3]

        # セルインデックス計算
        ix = np.clip((xs / cell_w).astype(int), 0, grid_W - 1)
        iy = np.clip((ys / cell_h).astype(int), 0, grid_H - 1)

        flat = iy * grid_W + ix
        
        buffer_deque = deque()
        for tsi, idx in zip(ts, flat):
            buffer_deque.append((tsi, idx))

        cnt_flat = np.bincount(flat, minlength=n_cells)
        count_mat = cnt_flat.reshape((grid_H, grid_W))

        # -----------ウィンドウを1msずつスライド--------------

        current_start = 0

        while True:
            next_start = current_start + step_time

            # eventsの古い1ms分を消す
            # while buffer_deque and buffer_deque[0][0] < next_start:
            #     tsi_old, idx_old = buffer_deque.popleft()
            #     cnt_flat[idx_old] -= 1

            # eventsに新しい1ms分を足す
            events = reader.load_delta_t(step_time)
            events = np.array(events.tolist(), dtype=int)

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
# wave_cluster.load_evt3()
# wave_cluster.sliding_buffer_count_matrices()
# wave_cluster.show_buffer_shapes()
# wave_cluster.visualize_count_matrices()
# wave_cluster.sum_and_visualize_interval()
# wave_cluster.apply_wavelet_transform()
# wave_cluster.process_streaming()
wave_cluster.test2()