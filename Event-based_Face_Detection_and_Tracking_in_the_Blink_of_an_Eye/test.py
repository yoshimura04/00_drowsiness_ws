import numpy as np
import cv2
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import DatWriter
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import csv
import bisect




class FrameProcessor:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        
        # 動画の読み込み
        self.cap = cv2.VideoCapture(self.input_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"動画ファイル {self.input_video_path} をオープンできませんでした。")
        
        # 動画のプロパティ取得
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # プロパティの表示
        print("FPS:", self.fps)
        print("Width:", self.width)
        print("Height:", self.height)
        
        # 出力動画の設定（MP4, mp4vエンコーディング）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.width, self.height))
        


        # 各フレームごとの目の座標を保存するリスト
        # 座標が検出できた場合は [(x1, y1), (x2, y2), ...]、検出できなければ [] となる
        self.eye_coords = []
    
    def annotate_video(self, max_num_faces=1):
        """動画を1フレームずつ処理して出力動画として保存する"""

        # MediaPipe Face Meshの初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=max_num_faces,
                                                    refine_landmarks=True)

        # 目のランドマークインデックスを右目と左目で分割
        self.left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, ]    # 左目：例として左外側、左内側
        self.right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]  # 右目：例として右外側、右内側

        # 右目と左目の座標リストを初期化（各フレームごとに座標を保存）
        self.left_eye_coords = []
        self.right_eye_coords = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 各フレームごとに左右の目の座標を保持するリストを初期化
            current_left_eye_coords = []
            current_right_eye_coords = []
            
            # BGR → RGB変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face Mesh 推論
            results = self.face_mesh.process(rgb_frame)
            
            # 目のランドマークが検出できた場合に描画
            if results.multi_face_landmarks:
                # ここでは最初の検出結果のみを利用（max_num_faces=1の場合）
                face_landmarks = results.multi_face_landmarks[0]
                
                # 左目の処理
                for idx in self.left_eye_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                    current_left_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)  # 緑色の点を描画
                    cv2.putText(frame, f"({x}, {y})", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                
                # 右目の処理
                for idx in self.right_eye_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * self.width)
                    y = int(landmark.y * self.height)
                    current_right_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)  # 緑色の点を描画
                    cv2.putText(frame, f"({x}, {y})", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

            # フレームごとに左右の目の座標リストに追加
            self.left_eye_coords.append(current_left_eye_coords)
            self.right_eye_coords.append(current_right_eye_coords)

            # 処理したフレームを出力動画に書き込み
            self.out.write(frame)
        
        print(f"left_eye_coords: {self.left_eye_coords}")
        print(f"right_eye_coords: {self.right_eye_coords}")
        print(f"フレーム数: {len(self.left_eye_coords)}")

        left_eye_array = np.array(self.left_eye_coords)
        print(f"left_eye_coords shape: {left_eye_array.shape}")
        
        self.release_resources()
    
    def release_resources(self):
        """リソースの解放"""
        self.cap.release()
        self.out.release()
        self.face_mesh.close()





class EventProcessor:
    def __init__(self,
                 # 密度フィルタ関連パラメータ
                 width=1280,
                 height=720,
                 radius=1,
                 high_density_threshold=60,
                 low_density_threshold=3,
                 density_threshold_switch=40,
                 max_density=210,
                 max_total_density=240,
                 time_threshold=2500,
                 # アクティビティ関連パラメータ
                 tau=50000.0,
                 scale=1.0,
                 tile_width=19,
                 tile_height=15,
                 img_width=1280,
                 img_height=720):
        # 密度フィルタ関連パラメータの設定
        self.width = width
        self.height = height
        self.radius = radius
        self.high_density_threshold = high_density_threshold
        self.low_density_threshold = low_density_threshold
        self.density_threshold_switch = density_threshold_switch
        self.max_density = max_density
        self.max_total_density = max_total_density
        self.time_threshold = time_threshold
        
        # アクティビティ関連パラメータの設定
        self.tau = tau
        self.scale = scale
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.img_width = img_width
        self.img_height = img_height
        
        # 画像をタイル分割する際のグリッドサイズを計算
        self.n_tiles_x = self.img_width // self.tile_width
        self.n_tiles_y = self.img_height // self.tile_height
        
        # アクティビティ履歴を格納するためのリスト
        self.activity_history_on = []
        self.activity_history_off = []
        self.time_history = []

        

    def load_evt3(self, file_path, time_window=10000):
        """
        EVT3ファイルを読み込み、NumPy配列として返すメソッド
        ※ RawReader クラスは事前に定義されている前提です。
        """
        reader = RawReader(file_path)
        self.event_data = []
        
        while not reader.is_done():
            events = reader.load_delta_t(time_window)
            if events is not None:
                self.event_data.append(events)
        
        self.event_data = np.concatenate(self.event_data)
        
        print(f"Loaded {len(self.event_data)} events.")
        print("First 10 events:")
        print(self.event_data[:10])
        print("Last 10 events:")
        print(self.event_data[-10:])
        print(f"len: {len(self.event_data)}")
        print(f"dtype: {self.event_data.dtype}")

        self.trigger_events = []

        try:
            self.trigger_events = reader.get_ext_trigger_events()
            print(f"trigger events: {self.trigger_events}")
            print(f"trigger events num: {len(self.trigger_events)}")
        except Exception as e:
            print("トリガーイベント取得中にエラーが発生しました:", e)
        
        return self.event_data

    def normalize_event_timestamps(self, event_data):
        """
        イベントデータのタイムスタンプを正規化するメソッド
        event_data: 構造化配列（各イベントは ('x', 'y', 'p', 't') のフィールドを持つ）
        """
        norm_events = event_data.copy()
        first_t = event_data[0][3]
        norm_events['t'] -= first_t

        print("First 10 norm_events:")
        print(norm_events[:10])
        print("Last 10 norm_events:")
        print(norm_events[-10:])
        print(f"norm_events type: {type(norm_events)}")

        return norm_events

    def filter_events_by_density(self, events):
        """
        NumPy配列形式のイベントを密度に基づいてフィルタリングするメソッド
        events: 各行は (x, y, polarity, ts) の形式
        出力: フィルタを通過したイベントを同じ形式の NumPy 配列として返す
        """
        # 各画素の密度と最新タイムスタンプを保持するマップを初期化
        density_map = np.zeros((self.height, self.width), dtype=np.int32)
        timestamp_map = np.zeros((self.height, self.width), dtype=np.int32)
        
        filtered_events = []
        
        for event in events:
            try:
                x = int(event['x'])
                y = int(event['y'])
                polarity = int(event['p'])
                ts = int(event['t'])
            except Exception as e:
                print(f"Skipping invalid event: {event}, error: {e}")
                continue
            
            # 座標が範囲外の場合はスキップ
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            current_density = density_map[y, x]
            last_ts = timestamp_map[y, x]
            
            # タイムスタンプ差により密度を更新
            if ts - last_ts > self.time_threshold:
                new_density = 1
            else:
                new_density = min(current_density + 1, self.max_density)
            density_map[y, x] = new_density
            timestamp_map[y, x] = ts
            
            # 近傍（中心画素を含む）の密度を計算
            neighborhood_density = 0
            for dx in range(-self.radius, self.radius + 1):
                for dy in range(-self.radius, self.radius + 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighborhood_density += density_map[ny, nx]
            neighborhood_density = min(neighborhood_density, self.max_total_density)
            
            # 近傍密度に応じた閾値の設定
            if neighborhood_density >= self.density_threshold_switch:
                density_threshold = self.high_density_threshold
            else:
                density_threshold = self.low_density_threshold
            
            # 閾値を超える場合、イベントを出力リストに追加
            if neighborhood_density >= density_threshold:
                filtered_events.append((x, y, polarity, ts))

        print(f"filtered_events has {len(filtered_events)} events.")
        
        return np.array(filtered_events) if filtered_events else np.empty((0, 4), dtype=int)

    def calculate_activity_history(self, event_data):
        """
        イベントデータを処理し、各グリッド（タイル）のアクティビティを更新するメソッド
        結果は、self.activity_history_on, self.activity_history_off, self.time_history に保存される
        event_data: 構造化配列（各イベントは ('x', 'y', 'p', 't') のフィールドを持つ）
        """
        # タイルごとのアクティビティを保持する配列（ON, OFF）
        activity_on = np.zeros((self.n_tiles_y, self.n_tiles_x))
        activity_off = np.zeros((self.n_tiles_y, self.n_tiles_x))
        # 各タイルの最終更新時刻（初期値は0）
        last_time = np.zeros((self.n_tiles_y, self.n_tiles_x))
        
        # 履歴リストを初期化
        self.activity_history_on = []
        self.activity_history_off = []
        self.time_history = []
        
        # イベントデータ内のユニークな時刻で処理
        unique_times = np.unique(event_data['t'])
        for t in unique_times:
            # 現在時刻 t のイベントを抽出
            events_at_t = event_data[event_data['t'] == t]
            
            # 各タイル（グリッド）ごとにアクティビティを更新
            for tile_y in range(self.n_tiles_y):
                for tile_x in range(self.n_tiles_x):
                    dt = t - last_time[tile_y, tile_x]
                    # 指数関数的減衰を適用
                    activity_on[tile_y, tile_x] *= np.exp(-dt / self.tau)
                    activity_off[tile_y, tile_x] *= np.exp(-dt / self.tau)
                    
                    # タイル領域の座標を算出
                    x_min = tile_x * self.tile_width
                    x_max = (tile_x + 1) * self.tile_width
                    y_min = tile_y * self.tile_height
                    y_max = (tile_y + 1) * self.tile_height
                    
                    # 現在時刻 t のイベントのうち、このタイルに属するものを抽出
                    in_tile = (events_at_t['x'] >= x_min) & (events_at_t['x'] < x_max) & \
                              (events_at_t['y'] >= y_min) & (events_at_t['y'] < y_max)
                    events_in_tile = events_at_t[in_tile]
                    
                    # タイル内の各イベントでアクティビティを更新
                    for event in events_in_tile:
                        if event['p'] == 1:  # ONイベント
                            activity_on[tile_y, tile_x] += 1 / self.scale
                        else:                # OFFイベント
                            activity_off[tile_y, tile_x] += 1 / self.scale
                    
                    # タイルの最終更新時刻を更新
                    last_time[tile_y, tile_x] = t
            
            # 現在時刻 t のスナップショットを保存
            self.activity_history_on.append(activity_on.copy())
            self.activity_history_off.append(activity_off.copy())
            self.time_history.append(t)

    def plot_activity_history(self):
        """
        各時刻における全タイルの平均アクティビティ値を計算し、
        ONイベントとOFFイベントの推移をグラフに表示するメソッド
        """
        on_mean_values = [np.mean(activity) for activity in self.activity_history_on]
        off_mean_values = [np.mean(activity) for activity in self.activity_history_off]
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, on_mean_values, marker='o', label='ON Activity Mean')
        plt.plot(self.time_history, off_mean_values, marker='s', label='OFF Activity Mean')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Activity')
        plt.title('Activity Transition Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_global_activity_history(self, event_data):
        """
        イベントデータを処理し、全体のアクティビティを更新するメソッド（タイル分割なし）
        結果は、self.activity_history_global_on, self.activity_history_global_off, self.time_history_global に保存される
        event_data: 構造化配列（各イベントは ('x', 'y', 'p', 't') のフィールドを持つ）
        """
        # グローバルなアクティビティの初期化 (ON, OFF)
        global_activity_on = 0.0
        global_activity_off = 0.0
        # 最終更新時刻の初期値
        last_time = 0.0

        # 履歴リストを初期化
        self.activity_history_global_on = []
        self.activity_history_global_off = []
        self.time_history_global = []

        # イベントデータ内のユニークな時刻で処理
        unique_times = np.unique(event_data['t'])
        for t in unique_times:
            # 経過時間 dt に対して指数関数的減衰を適用
            dt = t - last_time
            global_activity_on *= np.exp(-dt / self.tau)
            global_activity_off *= np.exp(-dt / self.tau)

            # 現在時刻 t のイベントを抽出
            events_at_t = event_data[event_data['t'] == t]

            # 現在時刻 t の各イベントでアクティビティを更新
            for event in events_at_t:
                if event['p'] == 1:  # ONイベントの場合
                    global_activity_on += 1 / self.scale
                else:              # OFFイベントの場合
                    global_activity_off += 1 / self.scale

            # 最終更新時刻を更新
            last_time = t

            # 現在時刻 t のスナップショットを保存
            self.activity_history_global_on.append(global_activity_on)
            self.activity_history_global_off.append(global_activity_off)
            self.time_history_global.append(t)

    def calculate_global_activity_history_fast(self, event_data):
        """
        イベントデータを処理し、全体のアクティビティを再帰的に更新するメソッド
        結果は、self.activity_history_global_on, self.activity_history_global_off, self.time_history_global に保存される
        """

        # イベントデータが空の場合、何も処理せず空のリストを設定して返す
        if event_data.size == 0:
            self.activity_history_global_on = []
            self.activity_history_global_off = []
            self.time_history_global = []
            return

        # ユニークな時刻と、その各イベントが属するグループのインデックスを取得
        unique_times, inv, counts = np.unique(event_data['t'], return_inverse=True, return_counts=True)
        
        # 各イベントがONかどうかのブール配列をfloatに変換
        p_is_on = (event_data['p'] == 1).astype(np.float64)
        
        # 各ユニーク時刻ごとのONイベント数を集計
        on_counts = np.bincount(inv, weights=p_is_on)
        # OFFイベント数は全体件数からONイベント数を引く
        off_counts = counts - on_counts
        
        # スケールで正規化
        a_on = on_counts / self.scale
        a_off = off_counts / self.scale

        n = unique_times.shape[0]
        
        # 再帰的にアクティビティを計算
        global_activity_on = np.empty(n)
        global_activity_off = np.empty(n)
        
        # 初期値
        global_activity_on[0] = a_on[0]
        global_activity_off[0] = a_off[0]
        
        for i in range(1, n):
            dt = unique_times[i] - unique_times[i - 1]
            decay = np.exp(-dt / self.tau)
            global_activity_on[i] = global_activity_on[i - 1] * decay + a_on[i]
            global_activity_off[i] = global_activity_off[i - 1] * decay + a_off[i]
        
        # 結果をメンバ変数に保存
        self.activity_history_global_on = global_activity_on.tolist()
        # print(f"self.activity_history_global_on: {self.activity_history_global_on}")
        self.activity_history_global_off = global_activity_off.tolist()
        self.time_history_global = unique_times.tolist()

        # print(f"activity_history_global_on: {self.activity_history_global_on}")

    def plot_global_activity_history(self):
        """
        各時刻における全体のアクティビティ（タイル分割なし）の推移をグラフに表示するメソッド
        グローバルなONアクティビティとOFFアクティビティをそれぞれプロットする
        """
        # グローバルなアクティビティはスカラー値なので、そのまま履歴リストから取得
        global_on_values = self.activity_history_global_on
        global_off_values = self.activity_history_global_off
        time_values = self.time_history_global

        plt.figure(figsize=(10, 6))
        plt.plot(time_values, global_on_values, label='Global ON Activity', linewidth=1)
        plt.plot(time_values, global_off_values, label='Global OFF Activity', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Activity')
        plt.title('Global Activity Transition Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    # def plot_global_activity_history(self):
    #     """
    #     各時刻における全体のアクティビティ（タイル分割なし）の推移をグラフに表示するメソッド
    #     グローバルなONアクティビティとOFFアクティビティをそれぞれプロットする
    #     """
    #     global_on_values = self.activity_history_global_on
    #     global_off_values = self.activity_history_global_off
    #     time_values = self.time_history_global

    #     # 配列の長さが一致しているか確認
    #     if not (len(time_values) == len(global_on_values) == len(global_off_values)):
    #         raise ValueError("time_history_global, activity_history_global_on, activity_history_global_off の長さが一致していません")

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(time_values, global_on_values, label='Global ON Activity', linewidth=1)
    #     plt.plot(time_values, global_off_values, label='Global OFF Activity', linewidth=1)
    #     plt.xlabel('Time')
    #     plt.ylabel('Activity')
    #     plt.title('Global Activity Transition Over Time')
    #     plt.legend()
    #     plt.grid(True)

    #     # 値が非常に小さい場合、デフォルトのスケールでは変化が見えにくいため、
    #     # y軸の範囲を手動で設定（ここではデータ範囲に少し余裕を持たせています）
    #     y_min = min(min(global_on_values), min(global_off_values))
    #     y_max = max(max(global_on_values), max(global_off_values))
    #     if abs(y_max - y_min) < 1e-5:
    #         plt.ylim(y_min - 1e-6, y_max + 1e-6)
    #     else:
    #         plt.ylim(y_min, y_max)

    #     plt.show()
    
    def save_csv_file(self, events, output_file_path):
        """
        処理後のイベントデータを CSV ファイルに保存するメソッド（ヘッダーは出力しない）

        Parameters:
        events: NumPy 配列。各イベントは ('x', 'y', 'p', 't') のフィールドを持つ構造化配列、
                または形状 (n,4) の通常の配列であることを想定。
        output_file_path: 保存先のファイルパス（例: "filtered_events.csv"）
        """
        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 各イベントを1行として書き込む（ヘッダーは出力しない）
            for event in events:
                # 構造化配列なら各フィールドの値を取得、通常の配列ならそのままリスト化
                if events.dtype.names is not None:
                    row = [event[field] for field in events.dtype.names]
                else:
                    row = list(event)
                writer.writerow(row)
        
        print(f"CSV file saved to {output_file_path}")
    
    def get_frame_event_data(self):
        """
        各フレームのイベントデータを取得する関数
        self.trigger_events: (type, timestamp, 0) のタプルで記録され、偶数番目が露光開始、奇数番目が露光終了と仮定
        self.event_data: (x, y, polarity, timestamp) のタプルで記録され、タイムスタンプ順にソート済みと仮定
        """
        # 各イベントのタイムスタンプのリストを作成
        timestamps = [event[3] for event in self.event_data]
        self.frame_events = []

        num_frames = len(self.trigger_events) // 2
        for i in range(num_frames):
            start_time = self.trigger_events[2 * i][1]
            end_time = self.trigger_events[2 * i + 1][1]

            # 二分探索で開始・終了インデックスを取得
            start_index = bisect.bisect_left(timestamps, start_time)
            end_index = bisect.bisect_left(timestamps, end_time)

            events_in_frame = self.event_data[start_index:end_index]
            self.frame_events.append(events_in_frame)
        
        # print(f"frame_events: {self.frame_events}")
        # print(f"frame_events num: {len(self.frame_events)}")

        return self.frame_events


    def filter_eye_events(self, left_eye_coords, right_eye_coords, margin=10):
        """
        インスタンス変数 self.frame_events, self.left_eye_coords, self.right_eye_coords を使用して、
        各フレームのイベントデータから左右の目の周辺のイベントのみを抽出し、
        全フレーム分を1つの NumPy 構造付き配列にまとめて、
        self.left_eye_events と self.right_eye_events に格納するメソッドです。

        前提:
        - self.frame_events:
            各要素は1フレーム分のイベントデータ（NumPyの構造付き配列）。
            dtype は [('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')] となっている。
        - self.left_eye_coords:
            各要素は1フレーム分の左目の端を表す2点 (x, y) タプルのリスト。
        - self.right_eye_coords:
            各要素は1フレーム分の右目の端を表す2点 (x, y) タプルのリスト。

        Parameters:
        margin: バウンディングボックスに追加する余裕（ピクセル単位）。デフォルトは10。
        """
        import numpy as np

        left_events_all = []   # 左目のフィルタ結果を一時的に格納するリスト
        right_events_all = []  # 右目のフィルタ結果を一時的に格納するリスト
        
        # 基本となるdtype（frameフィールドは不要）
        base_dtype = self.frame_events[0].dtype
        
        # 各フレームのイベントデータ、左目の座標、右目の座標を同時に処理
        for events, left_coords, right_coords in zip(self.frame_events, left_eye_coords, right_eye_coords):
            # 左目：与えられた2点から x, y の最小・最大値を求め、marginを追加
            left_x_vals = [pt[0] for pt in left_coords]
            left_y_vals = [pt[1] for pt in left_coords]
            left_x_min = min(left_x_vals) - margin
            left_x_max = max(left_x_vals) + margin
            left_y_min = min(left_y_vals) - margin
            left_y_max = max(left_y_vals) + margin

            # 右目：与えられた2点から x, y の最小・最大値を求め、marginを追加
            right_x_vals = [pt[0] for pt in right_coords]
            right_y_vals = [pt[1] for pt in right_coords]
            right_x_min = min(right_x_vals) - margin
            right_x_max = max(right_x_vals) + margin
            right_y_min = min(right_y_vals) - margin
            right_y_max = max(right_y_vals) + margin

            # イベントデータから x, y の値を抽出
            x = events['x']
            y = events['y']

            # バウンディングボックス内にあるイベントを抽出
            left_mask = (x >= left_x_min) & (x <= left_x_max) & (y >= left_y_min) & (y <= left_y_max)
            right_mask = (x >= right_x_min) & (x <= right_x_max) & (y >= right_y_min) & (y <= right_y_max)
            
            left_filtered = events[left_mask]
            right_filtered = events[right_mask]
            
            if left_filtered.size > 0:
                left_events_all.append(left_filtered)
            if right_filtered.size > 0:
                right_events_all.append(right_filtered)
        
        # 各フレームのフィルタ結果を1つの配列に結合
        if left_events_all:
            self.left_eye_events = np.concatenate(left_events_all)
        else:
            self.left_eye_events = np.empty(0, dtype=base_dtype)
        
        if right_events_all:
            self.right_eye_events = np.concatenate(right_events_all)
        else:
            self.right_eye_events = np.empty(0, dtype=base_dtype)

        print(f"left_eye_events: {self.left_eye_events}")
        print(f"left_eye_events num: {len(self.left_eye_events)}")
    
    def split_events(self, n, m):
        """
        画像のサイズ（self.width, self.height）に基づいて、イベントデータを n*m のグリッドに分割するインスタンスメソッド
        ベクトル化により、各イベントに対する個別処理を削減し、高速化を図ります。
        
        結果は self.grid に格納され、各セルには対応するイベントの numpy 配列が保存されます。
        """
        import numpy as np
        
        # セルサイズの計算
        x_bin_size = self.width / n
        y_bin_size = self.height / m

        # 全イベントに対してセルインデックスを一括計算（ベクトル化）
        x_idx = np.minimum((self.event_data['x'] / x_bin_size).astype(int), n - 1)
        y_idx = np.minimum((self.event_data['y'] / y_bin_size).astype(int), m - 1)

        # グリッド（n x m）の初期化
        self.grid = [[None for _ in range(m)] for _ in range(n)]
        
        # 各セルごとに、対応するイベントを一括抽出
        for i in range(n):
            for j in range(m):
                mask = (x_idx == i) & (y_idx == j)
                self.grid[i][j] = self.event_data[mask]
    
    def count_events_in_range(self, event_data, bin_width=1000):
        """
        指定された時間範囲（bin_width）ごとに、イベント数をカウントし、
        結果を構造化配列として self.event_counts_by_time に保存するメソッドです。
        
        イベントデータは (x, y, p, t) の形式で、t がタイムスタンプを表すと仮定します。
        event_data がリストまたは構造化配列のどちらにも対応します。
        
        Parameters:
            event_data: イベントデータ（リスト または numpy構造化配列）
            bin_width (int or float): 時間範囲の幅
        """
        # event_data がリストの場合、各イベントが (x, y, p, t) のタプルであると仮定
        if isinstance(event_data, list):
            times = np.array([e[3] for e in event_data])
        else:
            times = event_data['t']
        
        # イベントが存在しない場合は空の構造化配列を設定して終了
        if times.size == 0:
            dt = np.dtype([('t', times.dtype), ('count', 'i4')])
            self.event_counts_in_range = np.empty(0, dtype=dt)
            return

        # タイムスタンプの範囲を取得
        t_min = times.min()
        t_max = times.max()
        
        # 指定された bin_width でビン境界を作成（最後のビンにイベントが含まれるように t_max+bin_width まで）
        bins = np.arange(t_min, t_max + bin_width, bin_width)
        
        # np.histogram で各ビン内のイベント数を計算
        counts, bin_edges = np.histogram(times, bins=bins)
        
        # イベントが存在するビンのみ抽出（カウントが 0 のビンは除外）
        nonzero_mask = counts > 0
        nonzero_counts = counts[nonzero_mask]
        # 各ビンの左端の値を取得（これを時間として利用）
        bin_times = bin_edges[:-1][nonzero_mask]
        
        # 構造化配列のデータ型を定義（'t' は元のタイムスタンプの型、'count' は整数型）
        dt = np.dtype([('t', times.dtype), ('count', 'i4')])
        result = np.empty(nonzero_counts.shape[0], dtype=dt)
        result['t'] = bin_times
        result['count'] = nonzero_counts

        print(f"event_counts_in_range: {result}")
        
        # 結果をインスタンス変数に保存
        self.event_counts_in_range = result

    def plot_event_counts_in_range(self):
        """
        self.event_counts_in_range の各要素 (value, count) を用いてグラフを描画します。
        """
        # (値, カウント) のペアから x と y のリストを作成
        x_values = [pair[0] for pair in self.event_counts_in_range]
        y_values = [pair[1] for pair in self.event_counts_in_range]

        # グラフの設定
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', label="Event Counts")
        plt.xlabel("Event Value")
        plt.ylabel("Count")
        plt.title("Event Counts in Range")
        plt.legend()
        plt.grid(True)
        
        # グラフの表示
        plt.show()


    def extract_times_above_threshold_with_deadtime(self, threshold=25, dead_time_ms=500):
        """
        カウントが指定の閾値以上になったときの時間を抽出し、
        一度検出した後、dead_time_ms（ミリ秒）間は次の検出を行わないようにします。
        
        Parameters:
        -----------
        threshold : int
            カウントのしきい値
        dead_time_ms : int, optional
            検出後の無視期間（ミリ秒単位）、デフォルトは250ms
            
        Returns:
        --------
        list
            検出された時間（μs単位）のリスト
        """
        dead_time_us = dead_time_ms * 1000  # 250ms = 250,000μs
        self.blink_start_times = []
        last_detected_time = None

        for time, count in self.event_counts_in_range:
            if count >= threshold:
                # 最初の検出または前回検出からdead_time_us以上経過している場合
                if last_detected_time is None or (time - last_detected_time) >= dead_time_us:
                    self.blink_start_times.append(time)
                    last_detected_time = time

        print(f"blink_start_times: {self.blink_start_times}")
        return self.blink_start_times

    
    def save_activity_history_to_csv(self, filename):
        """
        self.activity_history_global_on, self.activity_history_global_off, self.time_history_global
        を取得して、指定されたCSVファイルに保存するメソッドです。
        
        CSVファイルの形式:
            time, global_activity_on, global_activity_off

        Parameters:
            filename (str): 保存先のCSVファイルのパス
        """
        # 3つのリストが同じ長さであることを確認
        if not (len(self.time_history_global) == len(self.activity_history_global_on) == len(self.activity_history_global_off)):
            raise ValueError("保存対象のリストの長さが一致していません。")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行の書き込み
            writer.writerow(["time", "global_activity_on", "global_activity_off"])
            # 各行に対して、対応する値をCSVに出力
            for t, on_val, off_val in zip(self.time_history_global, 
                                        self.activity_history_global_on, 
                                        self.activity_history_global_off):
                writer.writerow([t, on_val, off_val])

    def extract_activity_during_blinks(self, blink_duration_ms=250):
        """
        インスタンス変数 self.blink_start_times をもとに、各瞬きの期間（blink_duration_ms [ms]）中の
        時刻とアクティビティ値（global_on, global_off）を抽出し、結果をインスタンス変数 self.blink_activity_data に格納します。
        
        Parameters:
        -----------
        blink_duration_ms : int, optional
            瞬きの持続時間（ミリ秒単位）、デフォルトは250ms
            
        Returns:
        --------
        list of dict
            各瞬きごとに辞書形式で以下のキーを持つデータを返します。
            - "blink_start": 瞬き開始時刻（μs）
            - "blink_end": 瞬き終了時刻（μs）
            - "times": 瞬き期間中の時刻のリスト（μs）
            - "activity_on": 瞬き期間中の self.activity_history_global_on の値のリスト
            - "activity_off": 瞬き期間中の self.activity_history_global_off の値のリスト
        """
        # 瞬きの持続時間をμsに変換
        blink_duration_us = blink_duration_ms * 1000
        
        # 結果をインスタンス変数として初期化
        self.blink_activity_data = []
        
        # self.blink_start_times を利用して、各瞬きごとにデータを抽出
        for blink_start in self.blink_start_times:
            blink_end = blink_start + blink_duration_us
            
            # self.time_history_global から、blink_start から blink_end の間にあるインデックスを抽出
            indices = [i for i, t in enumerate(self.time_history_global) if blink_start <= t < blink_end]
            
            data = {
                "blink_start": blink_start,
                "blink_end": blink_end,
                "times": [self.time_history_global[i] for i in indices],
                "activity_on": [self.activity_history_global_on[i] for i in indices],
                "activity_off": [self.activity_history_global_off[i] for i in indices]
            }
            self.blink_activity_data.append(data)
        
        print(f"blink_activity_data: {self.blink_activity_data}")
        
        return self.blink_activity_data

    def plot_blink_activity_data(self):
        """
        self.blink_activity_data に格納された各瞬きのアクティビティデータをグラフにして表示します。
        
        各瞬きごとにサブプロットを作成し、瞬き期間中の時刻（μs）に対して
        activity_on と activity_off の値をプロットします。
        """
        # blink_activity_data が存在しない場合は、抽出を促す
        if not hasattr(self, "blink_activity_data") or not self.blink_activity_data:
            print("blink_activity_data が存在しません。抽出を先に実行してください。")
            return

        n_blinks = len(self.blink_activity_data)
        fig, axes = plt.subplots(n_blinks, 1, figsize=(10, 4 * n_blinks), sharex=False)

        # サブプロットが1つの場合、axesが単体となるためリストに変換
        if n_blinks == 1:
            axes = [axes]

        for ax, blink_data in zip(axes, self.blink_activity_data):
            times = blink_data.get("times", [])
            activity_on = blink_data.get("activity_on", [])
            activity_off = blink_data.get("activity_off", [])
            blink_start = blink_data.get("blink_start", None)
            blink_end = blink_data.get("blink_end", None)

            ax.plot(times, activity_on, marker='o', linestyle='-', label="Activity On")
            ax.plot(times, activity_off, marker='x', linestyle='--', label="Activity Off")
            ax.set_xlabel("Time (μs)")
            ax.set_ylabel("Activity Value")
            ax.set_title(f"Blink from {blink_start} to {blink_end} μs")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def compute_mean_blink_activity(self, num_samples=100):
        """
        各瞬きのデータ（self.blink_activity_data）を0〜1の正規化された時間軸に補間し、
        各正規化時刻における activity_on, activity_off の平均を計算して、結果を
        インスタンス変数 self.mean_blink_activity に格納します。

        Parameters:
        -----------
        num_samples : int, optional
            共通の正規化時間軸におけるサンプル数。デフォルトは100。

        Returns:
        --------
        dict
            'normalized_time': 0〜1の共通正規化時間軸
            'activity_on': 各正規化時刻における平均 activity_on 値
            'activity_off': 各正規化時刻における平均 activity_off 値
        """
        # 共通の正規化時間軸（0〜1の範囲）を作成
        common_time_norm = np.linspace(0, 1, num_samples)
        
        # 各瞬きの補間結果を格納するリスト
        interpolated_on = []
        interpolated_off = []
        
        # 各瞬きごとのデータに対して処理を行う
        for blink in self.blink_activity_data:
            times = np.array(blink["times"])
            # サンプルがない瞬きはスキップ
            if len(times) == 0:
                continue
            
            blink_start = blink["blink_start"]
            blink_end = blink["blink_end"]
            duration = blink_end - blink_start
            
            # 各瞬きのサンプル時刻を0〜1の正規化された時間に変換
            norm_times = (times - blink_start) / duration
            
            act_on = np.array(blink["activity_on"])
            act_off = np.array(blink["activity_off"])
            
            # 共通の正規化時間軸に対して線形補間
            interp_on = np.interp(common_time_norm, norm_times, act_on)
            interp_off = np.interp(common_time_norm, norm_times, act_off)
            
            interpolated_on.append(interp_on)
            interpolated_off.append(interp_off)
        
        # 各正規化時刻における各瞬きの値の平均を計算
        mean_on = np.mean(interpolated_on, axis=0)
        mean_off = np.mean(interpolated_off, axis=0)
        
        # 平均化結果をインスタンス変数に格納
        self.mean_blink_activity = {
            "normalized_time": common_time_norm,
            "activity_on": mean_on,
            "activity_off": mean_off
        }
        
        return self.mean_blink_activity

    def plot_mean_blink_activity(self):
        """
        self.mean_blink_activity に格納された平均化済み瞬きデータをグラフで表示します。
        x軸は正規化された時間 (0〜1)、y軸は activity_on と activity_off の値をプロットします。
        """
        # mean_blink_activity が存在するかチェック
        if not hasattr(self, "mean_blink_activity") or not self.mean_blink_activity:
            print("mean_blink_activity が存在しません。平均化処理を先に実行してください。")
            return

        mean_data = self.mean_blink_activity
        normalized_time = mean_data["normalized_time"]
        activity_on = mean_data["activity_on"]
        activity_off = mean_data["activity_off"]

        # グラフの作成
        plt.figure(figsize=(8, 6))
        plt.plot(normalized_time, activity_on, marker='o', linestyle='-', label="Activity On")
        plt.plot(normalized_time, activity_off, marker='x', linestyle='--', label="Activity Off")
        plt.xlabel("Normalized Time (0-1)")
        plt.ylabel("Activity Value")
        plt.title("Mean Blink Activity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_tile_activity(self, file_path, num_tiles_x=16, num_tiles_y=10):
        """
        指定された EVT3ファイルからイベントデータを読み込み、指定されたタイル数でイベントを分割し、
        各タイルのアクティビティを計算して、各セルのアクティビティを [activity_on, activity_off]
        の形で保持するインスタンス変数 self.tile_activity_history に格納します。
        
        Parameters:
        -----------
        file_path : str
            EVT3ファイル（イベントデータが記録されたファイル）のパス
        num_tiles_x : int, optional
            タイルの水平方向の数。デフォルトは16。
        num_tiles_y : int, optional
            タイルの垂直方向の数。デフォルトは16。
        
        Returns:
        --------
        self.tile_activity_history : list of list
            2次元リスト形式（num_tiles_x × num_tiles_y）で、各セルに [activity_on, activity_off] が格納される。
        """
        print("Loading EVT3 file...")
        # EVT3ファイルからイベントデータを読み込む（各イベントは (x, y, p, t) の形式を想定）
        self.load_evt3(file_path)

        # 指定されたタイル数でイベントを分割
        self.split_events(num_tiles_x, num_tiles_y)

        # インスタンス変数として tile_activity_history を 2次元リスト（各セルは [activity_on, activity_off]）
        # として宣言
        self.tile_activity_history = [[[0, 0, 0] for _ in range(num_tiles_y)] for _ in range(num_tiles_x)]

        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                # 各タイルに対してグローバルなアクティビティを計算
                self.calculate_global_activity_history_fast(self.grid[x][y])
                self.tile_activity_history[x][y][0] = self.activity_history_global_on
                self.tile_activity_history[x][y][1] = self.activity_history_global_off
                self.tile_activity_history[x][y][2] = self.time_history_global

                # イベント数をカウント（オプション）
                self.count_events_in_range(self.grid[x][y])
                print(f"event_counts_in_range: {self.event_counts_in_range}")

        return self.tile_activity_history

    def plot_all_tile_activity_histories(self):
        """
        self.tile_activity_history に格納された各タイルのアクティビティ履歴を、
        各タイルごとに個別の time_history_global, activity_on, activity_off を用いて、
        16×16 のサブプロットにプロットします。
        
        ※ 各サブプロットの x 軸（時刻）は全タイルで統一されます。
        ※ self.tile_activity_history の各セルは [activity_on, activity_off, time_history_global] の形式とします。
        """
        # タイル数の取得（self.tile_activity_history[x][y] 形式、x: 水平, y: 垂直）
        num_tiles_x = len(self.tile_activity_history)
        num_tiles_y = len(self.tile_activity_history[0]) if num_tiles_x > 0 else 0

        # 全タイルの time_history_global から x 軸の最小値・最大値を算出
        global_x_min, global_x_max = None, None
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                tile_data = self.tile_activity_history[x][y]
                tile_time = tile_data[2]  # 時系列データ
                if tile_time:
                    local_x_min = min(tile_time)
                    local_x_max = max(tile_time)
                    if global_x_min is None or local_x_min < global_x_min:
                        global_x_min = local_x_min
                    if global_x_max is None or local_x_max > global_x_max:
                        global_x_max = local_x_max

        # サブプロット作成（行: num_tiles_y, 列: num_tiles_x）
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x,
                                figsize=(num_tiles_x * 1.5, num_tiles_y * 1.5))
        
        # axes の形状が1次元の場合への対応
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]

        # サブプロットの各グラフにプロット（注意：サブプロットの行が y軸、列が x軸）
        for i in range(num_tiles_y):       # i: 行 (y軸)
            for j in range(num_tiles_x):   # j: 列 (x軸)
                ax = axes[i][j]
                # self.tile_activity_history は [x][y] 形式なので、データは [j][i]
                tile_data = self.tile_activity_history[j][i]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                
                if tile_time and tile_on and tile_off:
                    ax.plot(tile_time, tile_on, label='ON', linewidth=0.5)
                    ax.plot(tile_time, tile_off, label='OFF', linewidth=0.5)
                else:
                    ax.text(0.5, 0.5, "No Data",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)
                # x 軸の範囲を全タイルで統一
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(global_x_min, global_x_max)
                # y 軸は個別に設定（変更しない）

        fig.suptitle(f"Tile Activity Histories ({num_tiles_x}×{num_tiles_y}) with Unified X-Axis", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



    # xy軸の最大値と最小値を統一
    '''
    def plot_all_tile_activity_histories(self):
        """
        self.tile_activity_history に格納された各タイルのアクティビティ履歴を、
        各タイルごとに個別の time_history_global, activity_on, activity_off を用いて、
        サブプロットにプロットします。
        
        各グラフの x 軸と y 軸の範囲を全タイルで同じにします。
        
        前提:
        - self.tile_activity_history は 16×16 のリストで、
            各セルが [activity_on, activity_off, time_history_global] の形式になっている。
        - 各タイルの activity_on, activity_off, time_history_global はリスト形式の時系列データ。
        """
        # タイル数の取得（x軸が水平、y軸が垂直）
        num_tiles_x = len(self.tile_activity_history)
        num_tiles_y = len(self.tile_activity_history[0]) if num_tiles_x > 0 else 0

        # まず全タイルからグローバルな軸の最小値・最大値を求める
        global_x_min, global_x_max = None, None
        global_y_min, global_y_max = None, None

        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                tile_data = self.tile_activity_history[x][y]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                # データが存在する場合のみ処理
                if tile_time and tile_on and tile_off:
                    local_x_min = min(tile_time)
                    local_x_max = max(tile_time)
                    local_y_min = min(min(tile_on), min(tile_off))
                    local_y_max = max(max(tile_on), max(tile_off))
                    # グローバルな x 軸の更新
                    if global_x_min is None or local_x_min < global_x_min:
                        global_x_min = local_x_min
                    if global_x_max is None or local_x_max > global_x_max:
                        global_x_max = local_x_max
                    # グローバルな y 軸の更新
                    if global_y_min is None or local_y_min < global_y_min:
                        global_y_min = local_y_min
                    if global_y_max is None or local_y_max > global_y_max:
                        global_y_max = local_y_max

        # サブプロットの作成（注意：plt.subplots の引数は行数, 列数 なので、行数= num_tiles_y, 列数 = num_tiles_x）
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x, figsize=(num_tiles_x * 1.5, num_tiles_y * 1.5))

        # axes が2次元配列でない場合に対応
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]

        # 各タイルのグラフをプロット
        for i in range(num_tiles_y):       # i: 行 (y軸)
            for j in range(num_tiles_x):   # j: 列 (x軸)
                ax = axes[i][j]
                # タイルのデータは、元は self.tile_activity_history[x][y]、ここでは x = j, y = i にする
                tile_data = self.tile_activity_history[j][i]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                
                if tile_time and tile_on and tile_off:
                    ax.plot(tile_time, tile_on, label='ON', linewidth=0.5)
                    ax.plot(tile_time, tile_off, label='OFF', linewidth=0.5)
                else:
                    ax.text(0.5, 0.5, "No Data", horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)
                
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)
                
                # 軸の範囲を全タイルで統一
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(global_x_min, global_x_max)
                if global_y_min is not None and global_y_max is not None:
                    ax.set_ylim(global_y_min, global_y_max)

        fig.suptitle("Tile Activity Histories (16×16)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    '''


    # xy軸の最大値と最小値を統一しない

    '''
    def plot_all_tile_activity_histories(self):
        """
        self.tile_activity_history に格納された各タイルのアクティビティ履歴を、
        各タイルごとに個別の time_history_global, activity_on, activity_off を用いて、
        サブプロットにプロットします。

        前提:
        - self.tile_activity_history は 16×16 のリストで、
            各セルが [activity_on, activity_off, time_history_global] の形式になっている。
        - 各タイルの activity_on, activity_off, time_history_global はリスト形式の時系列データ。
        """
        # タイル数の取得（x軸が水平方向、y軸が垂直方向）
        num_tiles_x = len(self.tile_activity_history)
        num_tiles_y = len(self.tile_activity_history[0]) if num_tiles_x > 0 else 0

        # 修正：plt.subplots の引数を (行数, 列数) に合わせて、行数を num_tiles_y、列数を num_tiles_x にする
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x, figsize=(num_tiles_x * 1.5, num_tiles_y * 1.5))
        
        # axes が2次元配列でない場合（例: 1セルのみ）を考慮
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]

        # 各タイルのグラフをプロット（注意：行が y 軸、列が x 軸）
        for i in range(num_tiles_y):       # i: 行 (y軸)
            for j in range(num_tiles_x):   # j: 列 (x軸)
                ax = axes[i][j]
                # タイルのアクセス方法は、tile_activity_history[j][i] にする
                tile_data = self.tile_activity_history[j][i]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                
                if tile_time and tile_on and tile_off:
                    ax.plot(tile_time, tile_on, label='ON', linewidth=0.5)
                    ax.plot(tile_time, tile_off, label='OFF', linewidth=0.5)
                else:
                    ax.text(0.5, 0.5, "No Data", horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes)
                
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)

        fig.suptitle("Tile Activity Histories (16×16)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    '''

    def count_events_by_tile(self, bin_width=1000):
        """
        self.grid に格納された各タイルのイベントデータから、指定された時間範囲（bin_width）
        ごとにイベントをカウントし、結果（構造化配列）を各タイルごとに保存します。
        
        各タイルのカウント結果は self.event_counts_by_tile という 2 次元リストに保存され、
        各要素は count_events_in_range メソッドで算出された構造化配列となります。
        
        Parameters:
        -----------
        bin_width : int or float, optional
            時間範囲の幅（デフォルトは 1000）
        
        Returns:
        --------
        self.event_counts_by_tile : list of list
            self.grid と同じサイズの 2 次元リストで、各セルに対応する構造化配列が保存される。
        """
        # self.grid のサイズを取得
        n = len(self.grid)
        m = len(self.grid[0]) if n > 0 else 0

        # 結果を保存する 2 次元リストを初期化
        self.event_counts_by_tile = [[None for _ in range(m)] for _ in range(n)]

        # 各タイルのイベントデータに対して処理を実施
        for i in range(n):
            for j in range(m):
                tile_data = self.grid[i][j]
                # タイルごとに count_events_in_range を呼び出す
                self.count_events_in_range(tile_data, bin_width)
                # count_events_in_range は self.event_counts_in_range に結果を保存するので、
                # その内容をコピーして、対応するセルに格納する
                self.event_counts_by_tile[i][j] = self.event_counts_in_range.copy()

        return self.event_counts_by_tile

    def plot_event_counts_by_tile(self):
        """
        self.event_counts_by_tile に格納された各タイルのイベントカウント結果（構造化配列）を、
        16×16 のサブプロットとして一度に表示します。
        
        各サブプロットでは、x 軸に時刻 ('t')、y 軸にイベント数 ('count') を線のみでプロットし、
        すべてのグラフで x 軸の範囲を統一します（y 軸は各タイルごとに自動調整）。
        
        前提:
        - self.event_counts_by_tile は [x][y] の形式（self.grid と同じ順序）で保存され、
            各セルは count_events_in_range メソッドの結果（構造化配列）を保持している。
        - 各構造化配列はフィールド 't'（タイムスタンプ）と 'count'（イベント数）を持つ。
        """
        # タイル数の取得（self.event_counts_by_tile は [x][y] 形式）
        num_tiles_x = len(self.event_counts_by_tile)              # 水平方向（列）
        num_tiles_y = len(self.event_counts_by_tile[0]) if num_tiles_x > 0 else 0  # 垂直方向（行）

        # すべてのタイルからグローバルな x 軸（時刻）の最小・最大値を求める
        global_x_min, global_x_max = None, None

        for i in range(num_tiles_y):       # i: 行 (y軸)
            for j in range(num_tiles_x):   # j: 列 (x軸)
                tile_counts = self.event_counts_by_tile[j][i]
                if tile_counts.size != 0:
                    x_vals = tile_counts['t']
                    local_x_min = x_vals.min()
                    local_x_max = x_vals.max()
                    if global_x_min is None or local_x_min < global_x_min:
                        global_x_min = local_x_min
                    if global_x_max is None or local_x_max > global_x_max:
                        global_x_max = local_x_max

        # サブプロット作成（表示時は行数 = num_tiles_y, 列数 = num_tiles_x）
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x, figsize=(num_tiles_x * 2, num_tiles_y * 2))
        
        # axes の形状が1次元の場合への対応
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]
        
        # 各サブプロットにタイルのデータをプロット
        # self.event_counts_by_tile は [x][y] 形式なので、表示時は [j][i] とする
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                ax = axes[i][j]
                tile_counts = self.event_counts_by_tile[j][i]
                if tile_counts.size == 0:
                    ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                            ha="center", va="center")
                else:
                    x_vals = tile_counts['t']
                    y_vals = tile_counts['count']
                    # 線のみで表現（マーカーを指定しない）
                    ax.plot(x_vals, y_vals, linestyle='-', linewidth=1, label="Event Counts")
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.grid(True)
                # x 軸のみ全タイルで統一
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(global_x_min, global_x_max)

        fig.suptitle("Event Counts by Tile (Unified X-Axis, Line Only)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def extract_times_above_threshold_by_tile(self, threshold=25, dead_time_ms=500):
        """
        self.event_counts_by_tile に格納された各タイルのイベントカウント結果（構造化配列）
        から、指定された閾値 (threshold) を超えるイベントの時刻を抽出します。
        一度検出した後、dead_time_ms（ミリ秒）間は次の検出を行わないようにします。
        
        Parameters:
        -----------
        threshold : int, optional
            カウントの閾値（デフォルトは5）
        dead_time_ms : int, optional
            検出後の無視期間（ミリ秒単位、デフォルトは500ms）
        
        Returns:
        --------
        self.extracted_tile_times : list of list
            self.event_counts_by_tile と同じサイズの 2 次元リストで、各セルに該当タイルで閾値を超えた時刻のリストが保存される。
        """
        dead_time_us = dead_time_ms * 1000  # ミリ秒をμsに変換
        # self.event_counts_by_tile のサイズに合わせた 2 次元リストを初期化
        n = len(self.event_counts_by_tile)  # 水平方向（x方向）のタイル数
        m = len(self.event_counts_by_tile[0]) if n > 0 else 0  # 垂直方向（y方向）のタイル数
        self.extracted_tile_times = [[[] for _ in range(m)] for _ in range(n)]
        
        # 各タイルごとに処理
        for i in range(n):
            for j in range(m):
                tile_counts = self.event_counts_by_tile[i][j]
                times_above = []
                last_detected_time = None
                # tile_counts が空でない場合のみ処理
                if tile_counts.size != 0:
                    for pair in tile_counts:
                        t = pair['t']
                        count = pair['count']
                        if count >= threshold:
                            if last_detected_time is None or (t - last_detected_time) >= dead_time_us:
                                times_above.append(t)
                                last_detected_time = t
                self.extracted_tile_times[i][j] = times_above

        return self.extracted_tile_times

    def extract_activity_in_range_by_tile(self, window_duration=250000):
        """
        self.extracted_tile_times に格納された各タイルの抽出時刻から、
        対応するタイルのアクティビティ履歴（self.tile_activity_history）内の
        時刻、activity_on, activity_off のデータを、抽出時刻から window_duration までの範囲で取得します。
        
        各タイルごとに、抽出された各時刻に対して、以下の情報を辞書形式でまとめ、
        そのタイルの結果としてリストに格納します。
        
        各辞書の構造例:
        {
            'start': <抽出開始時刻>,
            'end': <抽出開始時刻 + window_duration>,
            'time': <該当範囲の時刻データ>,
            'activity_on': <該当範囲の ON アクティビティ値>,
            'activity_off': <該当範囲の OFF アクティビティ値>
        }
        
        結果は、self.extracted_activity_by_tile に 2 次元リスト（self.grid と同じ形状）として保存されます。
        
        Parameters:
        -----------
        window_duration : int or float, optional
            抽出する時間範囲の幅（μs単位、デフォルトは250,000 μs = 250 ms）
        
        Returns:
        --------
        self.extracted_activity_by_tile : list of list
            各タイルごとに抽出されたデータのリストを格納した 2 次元リスト
        """
        import numpy as np
        
        # self.extracted_tile_times と self.tile_activity_history のサイズを取得
        n = len(self.tile_activity_history)  # 水平方向のタイル数（x軸）
        m = len(self.tile_activity_history[0]) if n > 0 else 0  # 垂直方向のタイル数（y軸）
        
        # 結果を保存する 2 次元リストを初期化
        self.extracted_activity_by_tile = [[[] for _ in range(m)] for _ in range(n)]
        
        # 各タイルごとに処理
        for i in range(n):       # タイルの x 軸方向インデックス
            for j in range(m):   # タイルの y 軸方向インデックス
                # それぞれのタイルの抽出された時刻リスト
                tile_extracted_times = self.extracted_tile_times[i][j]
                # それぞれのタイルのアクティビティ履歴： [activity_on, activity_off, time_history_global]
                tile_activity = self.tile_activity_history[i][j]
                # 各データを numpy 配列に変換（万が一リストの場合に備える）
                activity_on = np.array(tile_activity[0])
                activity_off = np.array(tile_activity[1])
                tile_time = np.array(tile_activity[2])
                
                # このタイルの抽出結果を保存するリスト
                extracted_list = []
                
                # 抽出時刻がある場合のみ処理
                for start_time in tile_extracted_times:
                    # 抽出範囲の終了時刻
                    end_time = start_time + window_duration
                    # tile_time の中で start_time 以上 end_time 未満のインデックスを抽出
                    indices = np.where((tile_time >= start_time) & (tile_time < end_time))[0]
                    # 該当するデータがある場合、辞書にまとめる
                    if indices.size > 0:
                        extracted_dict = {
                            'start': start_time,
                            'end': end_time,
                            'time': tile_time[indices],
                            'activity_on': activity_on[indices],
                            'activity_off': activity_off[indices]
                        }
                        extracted_list.append(extracted_dict)
                # このタイルの結果を格納
                self.extracted_activity_by_tile[i][j] = extracted_list
                
        return self.extracted_activity_by_tile

    def plot_tile_activity_with_extraction_highlight(self):
        """
        self.tile_activity_history に格納された各タイルのアクティビティ履歴を、
        16×16 のサブプロットとして表示します。
        
        各サブプロットでは、x 軸に時刻（time_history_global）、
        y 軸にアクティビティ値（activity_on, activity_off）を線のみでプロットし、
        さらに self.extracted_activity_by_tile に格納された抽出区間を背景色でハイライトします。
        
        前提:
        - self.tile_activity_history は [x][y] 形式で、各セルは
            [activity_on, activity_off, time_history_global] の形式となっている。
        - self.extracted_activity_by_tile は同じ [x][y] 形式で、各セルは
            複数の辞書を含むリストになっており、各辞書は
            {'start': <抽出開始時刻>, 'end': <抽出終了時刻>, 'time': ..., 'activity_on': ..., 'activity_off': ...}
            の形式で抽出結果を保持している。
        - 各タイルの x 軸（時刻）は全タイルで統一されます（グローバルな最小／最大値で設定）。
        """
        # タイル数の取得（self.tile_activity_history は [x][y] 形式）
        num_tiles_x = len(self.tile_activity_history)
        num_tiles_y = len(self.tile_activity_history[0]) if num_tiles_x > 0 else 0

        # 全タイルからグローバルな x 軸（時刻）の最小・最大値を算出
        global_x_min, global_x_max = None, None
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                tile_data = self.tile_activity_history[x][y]
                tile_time = tile_data[2]  # 時系列データ
                if tile_time:
                    local_x_min = min(tile_time)
                    local_x_max = max(tile_time)
                    if global_x_min is None or local_x_min < global_x_min:
                        global_x_min = local_x_min
                    if global_x_max is None or local_x_max > global_x_max:
                        global_x_max = local_x_max

        # サブプロット作成（行数 = num_tiles_y, 列数 = num_tiles_x）
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x,
                                figsize=(num_tiles_x * 1.5, num_tiles_y * 1.5))
        
        # axes の形状が1次元の場合への対応
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]

        # 各タイルのグラフを描画（self.tile_activity_history は [x][y] 形式なので、表示時は [j][i] とする）
        for i in range(num_tiles_y):       # i: 行（y軸）
            for j in range(num_tiles_x):   # j: 列（x軸）
                ax = axes[i][j]
                # タイルのデータを取得
                tile_data = self.tile_activity_history[j][i]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                
                if tile_time and tile_on and tile_off:
                    # 基本のアクティビティ推移の線グラフ（線のみ）
                    ax.plot(tile_time, tile_on, linestyle='-', linewidth=0.5, label='ON')
                    ax.plot(tile_time, tile_off, linestyle='-', linewidth=0.5, label='OFF')
                else:
                    ax.text(0.5, 0.5, "No Data",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                
                # 抽出区間のハイライト（self.extracted_activity_by_tile は [x][y] 形式）
                if hasattr(self, 'extracted_activity_by_tile'):
                    extracted_segments = self.extracted_activity_by_tile[j][i]
                    for segment in extracted_segments:
                        start_time = segment['start']
                        end_time = segment['end']
                        # ax.axvspan で該当区間を半透明のオレンジ色でハイライト
                        ax.axvspan(start_time, end_time, color='orange', alpha=0.3)
                
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.grid(True)
                # x 軸の範囲を全タイルで統一
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(global_x_min, global_x_max)
                # y 軸は各タイルごとに自動設定

        fig.suptitle(f"Tile Activity Histories with Extracted Segments Highlighted", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


    def plot_tile_activity_with_extraction_highlight_with_correlation(self, correlation_by_tile):
        """
        self.tile_activity_history に格納された各タイルのアクティビティ履歴を
        16×16 のサブプロットに描画し、さらに self.extracted_activity_by_tile に格納された抽出区間を
        相関値に応じた透明度でハイライトします。
        
        Parameters:
        -----------
        correlation_by_tile : 2次元リスト
            各タイルごとに、抽出された各セグメントに対する相関値のリストが格納されている。
            (self.tile_activity_history, self.extracted_activity_by_tile と同じ [x][y] 形式)
            
        ※ self.tile_activity_history は [x][y] 形式で、各セルは
            [activity_on, activity_off, time_history_global] の形式で保持されている。
            self.extracted_activity_by_tile は同じ [x][y] 形式で、各セルは複数の辞書を含むリストとなっている。
        """
        # タイル数の取得（self.tile_activity_history は [x][y] 形式）
        num_tiles_x = len(self.tile_activity_history)
        num_tiles_y = len(self.tile_activity_history[0]) if num_tiles_x > 0 else 0

        # 全タイルからグローバルな x 軸（時刻）の最小・最大値を算出
        global_x_min, global_x_max = None, None
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                tile_data = self.tile_activity_history[x][y]
                tile_time = tile_data[2]  # 時系列データ
                if tile_time:
                    local_x_min = min(tile_time)
                    local_x_max = max(tile_time)
                    if global_x_min is None or local_x_min < global_x_min:
                        global_x_min = local_x_min
                    if global_x_max is None or local_x_max > global_x_max:
                        global_x_max = local_x_max

        # 全タイルの抽出区間の相関値からグローバルな最大相関値を求める
        global_max_corr = -np.inf
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                corr_list = correlation_by_tile[x][y]
                if corr_list:  # リストが空でなければ
                    local_max = max(corr_list)
                    if local_max > global_max_corr:
                        global_max_corr = local_max
        if global_max_corr == -np.inf:
            global_max_corr = 1.0  # もしデータがなければ 1.0 とする

        # ハイライト透明度の範囲
        min_alpha = 0.1
        max_alpha = 0.7

        # サブプロット作成（表示時は行数 = num_tiles_y, 列数 = num_tiles_x）
        fig, axes = plt.subplots(num_tiles_y, num_tiles_x,
                                figsize=(num_tiles_x * 1.5, num_tiles_y * 1.5))
        
        # axes の形状が1次元の場合への対応
        if num_tiles_y == 1 and num_tiles_x == 1:
            axes = [[axes]]
        elif num_tiles_y == 1:
            axes = [axes]
        elif num_tiles_x == 1:
            axes = [[ax] for ax in axes]
        
        # 各タイルのグラフを描画（注意：self.tile_activity_history は [x][y] 形式なので、表示時は [j][i] とする）
        for i in range(num_tiles_y):       # i: 行（y軸）
            for j in range(num_tiles_x):   # j: 列（x軸）
                ax = axes[i][j]
                # タイルのデータ取得
                tile_data = self.tile_activity_history[j][i]
                tile_on = tile_data[0]
                tile_off = tile_data[1]
                tile_time = tile_data[2]
                
                if tile_time and tile_on and tile_off:
                    # 基本のアクティビティ推移の線グラフ（線のみ）
                    ax.plot(tile_time, tile_on, linestyle='-', linewidth=0.5, label='ON')
                    ax.plot(tile_time, tile_off, linestyle='-', linewidth=0.5, label='OFF')
                else:
                    ax.text(0.5, 0.5, "No Data",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                
                # 抽出区間のハイライト：self.extracted_activity_by_tile は [x][y] 形式
                if hasattr(self, 'extracted_activity_by_tile'):
                    extracted_segments = self.extracted_activity_by_tile[j][i]
                    # 対応する相関値リスト（同じ順序と仮定）
                    corr_list = correlation_by_tile[j][i]
                    for idx, segment in enumerate(extracted_segments):
                        start_time = segment['start']
                        end_time = segment['end']
                        # 対応する相関値
                        corr_value = corr_list[idx] if idx < len(corr_list) else 0
                        # 相関値からハイライトの透明度を決定（global_max_corr を基準に正規化）
                        normalized = corr_value / global_max_corr if global_max_corr > 0 else 0
                        highlight_alpha = min_alpha + normalized * (max_alpha - min_alpha)
                        ax.axvspan(start_time, end_time, color='orange', alpha=highlight_alpha)
                
                ax.set_title(f"Tile ({j},{i})", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.grid(True)
                # x 軸の範囲を全タイルで統一
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(global_x_min, global_x_max)
        
        fig.suptitle("Tile Activity Histories with Correlation-Based Highlighting", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    

   
def convert_coordinates(img1_width, img1_height, img2_width, img2_height, coords):
    """
    画像1と画像2は同じ内容ですがサイズが異なります。
    画像1上の各フレームごとに格納された複数の座標 (coords は形状 (N, M, 2))
    を画像2上の対応する座標に変換します。
    
    Parameters:
        img1_width (int or float): 画像1の幅
        img1_height (int or float): 画像1の高さ
        img2_width (int or float): 画像2の幅
        img2_height (int or float): 画像2の高さ
        coords (list or ndarray): 画像1上の座標リスト。各要素はフレームごとの座標リストであり、
                                   各座標は [x, y] の形式です。
    
    Returns:
        list: 画像2上の座標リスト。各要素はフレームごとの座標リスト (各座標は [new_x, new_y] の形式) です。
    """
    converted = []
    for frame_coords in coords:
        frame_converted = []
        for coord in frame_coords:
            x, y = coord
            new_x = x * (img2_width / img1_width)
            new_y = y * (img2_height / img1_height)
            frame_converted.append([new_x, new_y])
        converted.append(frame_converted)
    return converted


def test():
    input_video_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250311_184103_126.mp4"
    output_video_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250311_184103_126_output.mp4"
    frame_processor = FrameProcessor(input_video_path, output_video_path)
    frame_processor.annotate_video()

    # EVT3ファイル（イベントデータが記録されたファイル）のパスを指定
    file_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250311_184103_126.raw"
    print("Loading EVT3 file...")
    
    # EventProcessor のインスタンスを生成（イベントデータの読み込み、正規化、フィルタ処理、アクティビティ計算などを行う）
    event_processor = EventProcessor()
    
    # EVT3ファイルからイベントデータを読み込む（各イベントは (x, y, p, t) の形式を想定）
    event_processor.load_evt3(file_path)
    
        
    # イベントのタイムスタンプを、最初のタイムスタンプを0にする形で正規化
    # norm_events = event_processor.normalize_event_timestamps(event_data)

    # 正規化後のイベントデータのうち、先頭100000イベントのみを使用（例）
    # event_processor.event_data = event_processor.event_data[:250000]
    # print(f"Norm_events has {len(norm_events)} events.")

    # filtered_events = event_processor.filter_events_by_density(norm_events)

    # event_processor.save_csv_file(event_data, "event_data.csv")

    # アクティビティ計算：各タイル（グリッド）ごとのアクティビティを算出し、履歴に保存
    # event_processor.calculate_global_activity_history_fast(event_data)
    
    # アクティビティ履歴のプロット（ON, OFF の各タイルの平均値の推移）
    # event_processor.plot_global_activity_history()

    event_processor.get_frame_event_data()

    # 目の座標をイベントの座標に変換
    converted_left_eye_coords = convert_coordinates(frame_processor.width, frame_processor.height, event_processor.width, event_processor.height, frame_processor.left_eye_coords)
    converted_right_eye_coords = convert_coordinates(frame_processor.width, frame_processor.height, event_processor.width, event_processor.height, frame_processor.right_eye_coords)

    event_processor.filter_eye_events(converted_left_eye_coords, converted_right_eye_coords)
    csv_file_name = "test.csv"
    event_processor.save_csv_file(event_processor.left_eye_events, csv_file_name)

    event_processor.calculate_global_activity_history_fast(event_processor.left_eye_events)
    event_processor.plot_global_activity_history()


def create_blink_model():
    input_video_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260.mp4"
    output_video_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260_output.mp4"
    frame_processor = FrameProcessor(input_video_path, output_video_path)
    frame_processor.annotate_video()

    # EVT3ファイル（イベントデータが記録されたファイル）のパスを指定
    file_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260.raw"
    print("Loading EVT3 file...")

    # EventProcessor のインスタンスを生成（イベントデータの読み込み、正規化、フィルタ処理、アクティビティ計算などを行う）
    event_processor = EventProcessor()

    # EVT3ファイルからイベントデータを読み込む（各イベントは (x, y, p, t) の形式を想定）
    event_processor.load_evt3(file_path)

    event_processor.get_frame_event_data()

    # 目の座標をイベントの座標に変換
    converted_left_eye_coords = convert_coordinates(frame_processor.width, frame_processor.height, event_processor.width, event_processor.height, frame_processor.left_eye_coords)
    converted_right_eye_coords = convert_coordinates(frame_processor.width, frame_processor.height, event_processor.width, event_processor.height, frame_processor.right_eye_coords)

    event_processor.filter_eye_events(converted_left_eye_coords, converted_right_eye_coords)

    # left
    event_processor.calculate_global_activity_history_fast(event_processor.left_eye_events)
    event_processor.plot_global_activity_history()

    filename = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260_left.csv"
    event_processor.save_activity_history_to_csv(filename)

    event_processor.count_events_in_range(event_processor.left_eye_events)
    event_processor.plot_event_counts_in_range()

    event_processor.extract_times_above_threshold_with_deadtime()
    # blink_start_times: [2936472, 4947472, 6965472, 8787472, 10722472]

    event_processor.extract_activity_during_blinks()
    event_processor.plot_blink_activity_data()

    event_processor.compute_mean_blink_activity()
    event_processor.plot_mean_blink_activity()
    left_mean_blink_activity = event_processor.mean_blink_activity

    # right
    event_processor.calculate_global_activity_history_fast(event_processor.right_eye_events)
    event_processor.plot_global_activity_history()

    filename = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250321_143628_260_right.csv"
    event_processor.save_activity_history_to_csv(filename)

    event_processor.count_events_in_range(event_processor.right_eye_events)
    event_processor.plot_event_counts_in_range()

    event_processor.extract_times_above_threshold_with_deadtime()
    # blink_start_times: [2937300, 4947300, 6963300, 8795300, 10727300]

    event_processor.extract_activity_during_blinks()
    event_processor.plot_blink_activity_data()

    event_processor.compute_mean_blink_activity()
    event_processor.plot_mean_blink_activity()
    right_mean_blink_activity = event_processor.mean_blink_activity

    return left_mean_blink_activity, right_mean_blink_activity


def compute_blink_correlation_by_tile(extracted_segments_2d, blink_model, alpha=2/3):
    """
    2次元配列 extracted_segments_2d（例：self.extracted_activity_by_tile）
    に格納された各タイルの抽出セグメントと、瞬きモデル blink_model との相関値を計算します。
    
    各抽出セグメント（辞書形式）は以下のキーを持つ前提です：
      'start'         : 抽出開始時刻（μs）
      'end'           : 抽出終了時刻（μs）
      'time'          : セグメント内の時刻データ（1次元 numpy array またはリスト）
      'activity_on'   : セグメント内の ON アクティビティ値
      'activity_off'  : セグメント内の OFF アクティビティ値
      
    blink_model は以下のキーを持つ辞書とします：
      'normalized_time': 0〜1 の正規化時間軸（1次元 numpy array）
      'activity_on'    : 瞬きモデルの ON 部分の値（1次元 array）
      'activity_off'   : 瞬きモデルの OFF 部分の値（1次元 array）
    
    Parameters:
    -----------
    extracted_segments_2d : list of list
        各セルに抽出されたセグメント（辞書）のリストが格納された2次元配列。
    blink_model : dict
        瞬きモデルを保持する辞書。
    alpha : float, optional
        ON と OFF の寄与を調整するパラメータ（デフォルトは 2/3）。
    
    Returns:
    --------
    correlation_by_tile : list of list
        各タイルに対応する、抽出セグメントごとの相関値のリストを保持する2次元配列。
    """
    n = len(extracted_segments_2d)  # 例: タイルの水平方向数
    m = len(extracted_segments_2d[0]) if n > 0 else 0  # 垂直方向数
    correlation_by_tile = [[[] for _ in range(m)] for _ in range(n)]
    
    # blink_model の情報
    norm_time_model = blink_model['normalized_time']
    model_on = blink_model['activity_on']
    model_off = blink_model['activity_off']
    
    # 各タイルごとにループ
    for i in range(n):
        for j in range(m):
            tile_segments = extracted_segments_2d[i][j]  # 各セルは抽出されたセグメントのリスト
            corr_list = []  # このタイル内の各セグメントに対する相関値
            for seg in tile_segments:
                t0 = seg['start']
                t1 = seg['end']
                window_duration = t1 - t0
                
                # セグメント内の時刻データを numpy 配列に変換し正規化（0～1）
                seg_time = np.array(seg['time'])
                norm_seg_time = (seg_time - t0) / window_duration
                
                # セグメント内のアクティビティ値
                seg_on = np.array(seg['activity_on'])
                seg_off = np.array(seg['activity_off'])
                
                # blink_model の正規化時間軸に合わせて線形補間
                interp_on = np.interp(norm_time_model, norm_seg_time, seg_on)
                interp_off = np.interp(norm_time_model, norm_seg_time, seg_off)
                
                # 論文の式に基づく相関値の計算
                C_on = np.sum(interp_on * model_on)
                C_off = np.sum(interp_off * model_off)
                C = alpha * C_on + (1 - alpha) * C_off
                corr_list.append(C)
            correlation_by_tile[i][j] = corr_list
    
    print(f"correlation_by_tile: {correlation_by_tile}")
            
    return correlation_by_tile


def main():
    left_mean_blink_activity, right_mean_blink_activity = create_blink_model()
    
    # 各タイルのアクティビティ値の推移から瞬きしてそうな箇所を抽出する
    test_data_processor = EventProcessor()
    file_path = "/home/carrobo2024/Downloads/BothView_V4_00/BothView/video/recording_250311_184103_126.raw"
    test_data_processor.calculate_tile_activity(file_path)
    test_data_processor.plot_all_tile_activity_histories()
    test_data_processor.count_events_by_tile()
    test_data_processor.plot_event_counts_by_tile()
    test_data_processor.extract_times_above_threshold_by_tile()
    test_data_processor.extract_activity_in_range_by_tile()
    test_data_processor.plot_tile_activity_with_extraction_highlight()

    # 瞬きモデルとの相関値を計算
    correlation_by_tile = compute_blink_correlation_by_tile(test_data_processor.extracted_activity_by_tile, left_mean_blink_activity)
    test_data_processor.plot_tile_activity_with_extraction_highlight_with_correlation(correlation_by_tile)

if __name__ == "__main__":
    main()
    


