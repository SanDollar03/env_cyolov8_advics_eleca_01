import cv2
import time
import os
import glob
from ultralytics import YOLO

# YOLOv8モデルをロード
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
model = YOLO(model_path)  # detect_live.py と同じフォルダのファイルを読み込む

# カメラキャプチャを初期化（デバイスIDが2のカメラを使用）
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# 現在のスクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# 結果を保存するフォルダを確認
output_folder = os.path.join(script_dir, 'data')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# dataフォルダを空にする関数
def clear_data_folder(folder):
    files = glob.glob(os.path.join(folder, '*.jpg'))
    for f in files:
        os.remove(f)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # YOLOv8でフレームを分析
    results = model(frame)

    # 「cell phone」を検知したかどうかを判定するフラグ
    detected_cell_phone = False

    # 分析結果を描画
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # "cell phone" を検知した場合、フラグをTrueに設定
            if model.names[int(cls)] == "cell phone":
                detected_cell_phone = True

    # ファイル名を決定
    output_filename = '1.jpg' if detected_cell_phone else '0.jpg'
    output_path = os.path.join(output_folder, output_filename)

    # 保存前にdataフォルダを空にする
    clear_data_folder(output_folder)

    # 分析結果をファイルに保存
    cv2.imwrite(output_path, frame)

    # 結果を表示
    cv2.imshow('YOLOv8 Live Detection', frame)

    # 1秒ごとにフレームを分析
    time.sleep(1)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
