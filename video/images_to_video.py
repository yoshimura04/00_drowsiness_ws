import os
import glob
import cv2
from tqdm import tqdm

def images_to_video(
    image_folder: str,
    output_path: str,
    fps: int = 30,
    ext: str = 'png'   # 'jpg', 'bmp' などにも変更可
):
    # 1. ファイルリスト取得・ソート
    pattern = os.path.join(image_folder, f'*.{ext}')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found with pattern: {pattern}")

    # 2. 動画サイズを決める（最初の画像を参照）
    first = cv2.imread(files[0])
    if first is None:
        raise ValueError(f"Cannot read image: {files[0]}")
    height, width = first.shape[:2]

    # 3. VideoWriter 初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # 4. 各画像を書き込む
    for img_path in tqdm(files, desc="Writing frames", unit="frame"):
        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"Warning: skip unreadable {img_path}")
            continue
        # サイズが揃っていない場合はリサイズ
        if img.shape[0:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        writer.write(img)

    # 5. 後始末
    writer.release()
    print(f"Saved video: {output_path} ({len(files)} frames @ {fps} fps)")


if __name__ == '__main__':
    images_to_video(
        image_folder='/home/carrobo2024/00_drowsiness_ws/video/dynamic_objects_in_the_background/test',
        output_path='/home/carrobo2024/00_drowsiness_ws/video/dynamic_objects_in_the_background/test/clusters_video.mp4',
        fps=30,
        ext='png'
    )