import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# conda : conda activate mask_eyes

class MediaPipeEyeDrawer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # 左右の目まわりランドマーク番号
        self.LEFT_EYE_IDX  = [33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        self.RIGHT_EYE_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

    def draw_eye_points_in_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレーム上に左右の目ランドマーク点を描画して返します。
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        results = self.face_mesh.process(image=rgb)

        out = frame.copy()
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # 左右の目ランドマークをそれぞれプロット
            for idx_list, color in [(self.LEFT_EYE_IDX, (0,255,0)),  # 左目：緑
                                     (self.RIGHT_EYE_IDX, (0,0,255))]: # 右目：赤
                for idx in idx_list:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    cv2.circle(out, (x,y), radius=2, color=color, thickness=-1)
                
                pts = np.array([
                    [int(lm[idx].x * w), int(lm[idx].y * h)]
                    for idx in idx_list
                ], dtype=np.int32)
                print(pts)
                print(type(pts), pts.dtype, pts.shape, pts.flags['C_CONTIGUOUS'])
        return out

    def mask_eyes_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        目領域を黒 (0)、それ以外を白 (255) にしたマスク画像を返します。
        """
        h, w, _ = frame.shape
        # MediaPipe に渡すための RGB 変換
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        results = self.face_mesh.process(image=rgb)

        # 全体を白で初期化
        mask = np.full((h, w), 255, dtype=np.uint8)
        if not results.multi_face_landmarks:
            # 顔が検出されなければ白背景マスクを返す
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 顔ランドマークを取得
        lm = results.multi_face_landmarks[0].landmark
        # 各目ポリゴンを黒で塗る
        for idx_list in (self.LEFT_EYE_IDX, self.RIGHT_EYE_IDX):
            pts = np.array([
                (int(lm[idx].x * w), int(lm[idx].y * h))
                for idx in idx_list
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)

        # グレースケールを BGR に変換して返す
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def draw_eye_points_in_video(self, in_path: str, out_path: str, codec='mp4v'):
        """
        動画中の各フレームに目ランドマーク点を描画し、出力動画を生成します。
        """
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {in_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Input video has {total_frames} frames.")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # img = self.draw_eye_points_in_frame(frame)
            img = self.mask_eyes_frame(frame)

            # 左右反転
            img = cv2.flip(img, 1)

            # print(type(img))
            writer.write(img)

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        print(f"Output saved to: {out_path}")

if __name__ == "__main__":
    drawer = MediaPipeEyeDrawer()
    # 画像テスト
    # img = cv2.imread("input.jpg")
    # out_img = drawer.draw_eye_points_in_frame(img)
    # cv2.imwrite("output_with_eye_points.jpg", out_img)

    # 動画テスト
    drawer.draw_eye_points_in_video(
        "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/recording_250708_224544_288.mp4",
        "/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/output_video_with_eye_points.mp4"
    )

# no_dynamic_objects : 373 frames