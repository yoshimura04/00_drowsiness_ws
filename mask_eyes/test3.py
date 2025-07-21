import cv2
import numpy as np
import mediapipe as mp

class MediaPipeEyeMasker:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5)
        # 左右の目まわりランドマーク番号
        self.LEFT_EYE_IDX  = [33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        self.RIGHT_EYE_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

    def mask_eyes_in_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        # results = self.face_mesh.process(rgb)
        results = self.face_mesh.process(image=rgb)
        mask = np.zeros((h, w), dtype=np.uint8)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            for idx_list in (self.LEFT_EYE_IDX, self.RIGHT_EYE_IDX):
                pts = np.array([(int(lm[i].x * w), int(lm[i].y * h))
                                for i in idx_list], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                pts = np.ascontiguousarray(pts)
                print(type(mask))
                print("===============================")
                cv2.fillPoly(mask, [pts], 255)

        masked = frame.copy()
        masked[mask==255] = (0,0,0)
        return masked

    def mask_video(self, in_path: str, out_path: str, codec='mp4v'):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画が開けません: {in_path}")
        fps    = cap.get(cv2.CAP_PROP_FPS)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
        while True:
            ret, frame = cap.read()
            if not ret: break
            masked = self.mask_eyes_in_frame(frame)
            writer.write(masked)
        cap.release(); writer.release()
        print(f"マスク済み動画を保存: {out_path}")

if __name__ == "__main__":
    masker = MediaPipeEyeMasker()
    # 画像テスト
    # img = cv2.imread("input.jpg")
    # out = masker.mask_eyes_in_frame(img)
    # cv2.imwrite("masked.jpg", out)
    # 動画テスト
    masker.mask_video("/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/recording_250708_224544_288.mp4", "masked.mp4")
