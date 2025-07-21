import cv2
import numpy as np
import os

class EyeMasker:
    def __init__(self,
                 face_cascade_path: str = None,
                 eye_cascade_path: str = None):
        """
        HaarCascade モデルをロードして初期化します。
        
        :param face_cascade_path: 顔検出用 XML のパス。None の場合は OpenCV の標準を使用。
        :param eye_cascade_path: 目検出用 XML のパス。None の場合は OpenCV の標準を使用。
        """
        cascade_dir = cv2.data.haarcascades
        if face_cascade_path is None:
            face_cascade_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
        if eye_cascade_path is None:
            eye_cascade_path = os.path.join(cascade_dir, 'haarcascade_eye.xml')
        
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)
    
    def mask_eyes_in_image(self, image: np.ndarray) -> np.ndarray:
        """
        与えられた BGR 画像から目領域を検出し、黒く塗りつぶした画像を返します。
        
        :param image: 入力 BGR 画像 (numpy.ndarray)
        :return: 目領域がマスクされた BGR 画像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(80,80))
        mask = np.zeros_like(image)
        
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(roi_gray,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30,30))
            for (ex, ey, ew, eh) in eyes:
                x1, y1 = fx + ex, fy + ey
                x2, y2 = x1 + ew, y1 + eh
                mask[y1:y2, x1:x2] = 255
        
        result = image.copy()
        result[mask[:,:,0] == 255] = (0, 0, 0)
        return result
    
    def mask_eyes_in_video(self,
                            input_path: str,
                            output_path: str,
                            codec: str = 'mp4v'):
        """
        動画からフレームごとに目領域を検出し、黒マスクをかけた動画を出力します。
        
        :param input_path: 入力動画ファイルへのパス
        :param output_path: 出力動画ファイルへのパス
        :param codec: 出力の FourCC コーデック（'mp4v', 'XVID' など）
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
    masker = EyeMasker()
    # 画像の例
    # img = cv2.imread("input.jpg")
    # out_img = masker.mask_eyes_in_image(img)
    # cv2.imwrite("output.jpg", out_img)

    print("DDD")
    
    # 動画の例
    masker.mask_eyes_in_video("/home/carrobo2024/00_drowsiness_ws/video/no_dynamic_objects_in_the_background/recording_250708_224544_288.mp4", "masked.mp4")
