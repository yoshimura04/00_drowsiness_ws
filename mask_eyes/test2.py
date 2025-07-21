import os
import cv2
import numpy as np
from typing import List, Tuple

# OpenPose Python API のパスを通す
# 例: sys.path.append('/path/to/openpose/python')
from openpose import pyopenpose as op

class OpenPoseEyeMasker:
    def __init__(self,
                 model_folder: str = "models/",
                 face: bool = True):
        """
        OpenPose を初期化。フェイシャルランドマークを出力する設定になっていること。
        
        :param model_folder: OpenPose のモデルファイル格納フォルダ
        :param face: フェイスキー点を使う場合は True
        """
        params = {
            "model_folder": model_folder,
            "face": 1 if face else 0,
            # 目だけなら body=0、hand=0 にして軽量化してもよい
            "body": 0,
            "hand": 0,
            "net_resolution": "320x240"
        }
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

    def _get_eye_landmarks(self, datum) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
        """
        OpenPose の datum.faceKeypoints から左右の目ランドマークを抽出。
        68点モデルなら
          - 左目: points 36–41
          - 右目: points 42–47
        """
        face_kps = datum.faceKeypoints  # shape (1, N, 3)
        if face_kps is None or face_kps.shape[1] < 48:
            return [], []

        pts = face_kps[0,:, :2].astype(int)  # (N,2)
        left_eye_pts  = pts[36:42].tolist()
        right_eye_pts = pts[42:48].tolist()
        return left_eye_pts, right_eye_pts

    def mask_eyes_in_image(self, img: np.ndarray) -> np.ndarray:
        """
        フレーム内の左右の目領域を convex hull で囲んで黒く塗りつぶす。
        """
        # OpenPose へ渡す
        datum = op.Datum()
        datum.cvInputData = img
        self.op_wrapper.emplaceAndPop([datum])

        left_eye, right_eye = self._get_eye_landmarks(datum)
        mask = np.zeros_like(img)

        for eye_pts in (left_eye, right_eye):
            if len(eye_pts) >= 3:
                hull = cv2.convexHull(np.array(eye_pts, dtype=np.int32))
                cv2.fillConvexPoly(mask, hull, (255,255,255))

        result = img.copy()
        # mask の白部分を黒く
        result[mask[:,:,0] == 255] = (0,0,0)
        return result

    def mask_eyes_in_video(self,
                            input_path: str,
                            output_path: str,
                            codec: str = 'mp4v'):
        """
        動画全体に対してマスク処理を適用して保存。
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"動画が開けません: {input_path}")
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            masked = self.mask_eyes_in_image(frame)
            writer.write(masked)

        cap.release()
        writer.release()
        print(f"マスク済み動画を保存しました: {output_path}")


if __name__ == "__main__":
    masker = OpenPoseEyeMasker(model_folder="path/to/openpose/models/")
    # 単一画像の例
    img = cv2.imread("input.jpg")
    out = masker.mask_eyes_in_image(img)
    cv2.imwrite("masked_image.jpg", out)
    # 動画の例
    masker.mask_eyes_in_video("input.mp4", "masked_video.mp4")
