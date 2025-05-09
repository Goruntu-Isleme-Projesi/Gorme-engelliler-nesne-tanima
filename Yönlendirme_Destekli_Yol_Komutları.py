from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# Ses motoru ve kilit sistemi
speak_lock = threading.Lock()

def speak(text):
    def run():
        with speak_lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    threading.Thread(target=run).start()

# Modeli yükle
model = YOLO('yolov8n.pt')

# Kamerayı aç
url = "http://192.168.1.17:4747/video"
cap = cv2.VideoCapture(url)

# Zaman kontrolü
last_spoken_time = 0
speak_interval = 3  # saniye

# Nesne gerçek genişlikleri (metre cinsinden)
real_widths = {
    'person': 0.5,
    'mouse': 0.07,
    'cell phone': 0.07,
    'car': 2.0,
    'dog': 0.5,
    'chair': 0.5,
    'bottle': 0.1,
    'laptop': 0.3
}

# Kamera odak uzunluğu
focal_length = 615

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, _ = frame.shape

    left_boundary = width // 3
    right_boundary = 2 * width // 3

    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False, device=0)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        classes = boxes.cls.cpu().numpy().astype(int)

        # Her bölge için engel var mı kontrolü
        left_blocked = False
        center_blocked = False
        right_blocked = False

        current_time = time.time()

        for cls_idx, box in zip(classes, boxes.xyxy):
            label = names[cls_idx]
            x1, y1, x2, y2 = box.cpu().numpy()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            pixel_width = x2 - x1
            real_width = real_widths.get(label, 0.5)

            if pixel_width > 0:
                distance = (real_width * focal_length) / pixel_width
            else:
                distance = -1

            # Sol-Orta-Sağ tespiti + engel kontrolü (yakınlık 2 metre altı)
            if distance != -1 and distance < 2.0:
                if center_x < left_boundary:
                    left_blocked = True
                elif center_x < right_boundary:
                    center_blocked = True
                else:
                    right_blocked = True

        # Sesli uyarı zamanı geldiyse
        if current_time - last_spoken_time > speak_interval:
            if center_blocked:
                speak("STOP!")
            elif left_blocked and not right_blocked:
                speak("TURN RİGHT!")
            elif right_blocked and not left_blocked:
                speak("TURN LEFT!")
            else:
                # Engel yoksa normal nesne bilgisi verelim
                for cls_idx, box in zip(classes, boxes.xyxy):
                    label = names[cls_idx]
                    x1, y1, x2, y2 = box.cpu().numpy()

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Hangi bölgede?
                    if center_x < width / 3:
                        horiz = "sol"
                    elif center_x < 2 * width / 3:
                        horiz = "orta"
                    else:
                        horiz = "sağ"

                    if center_y < height / 3:
                        vert = "üst"
                    elif center_y < 2 * height / 3:
                        vert = "orta"
                    else:
                        vert = "alt"

                    position = f"{horiz} {vert}"

                    pixel_width = x2 - x1
                    real_width = real_widths.get(label, 0.5)

                    if pixel_width > 0:
                        distance = (real_width * focal_length) / pixel_width
                        speak(f"{position} bölgede {label} algılandı, yaklaşık {distance:.1f} metre uzakta.")
                    else:
                        speak(f"{position} bölgede {label} algılandı.")

                    break  # sadece bir nesne için konuşsun

            last_spoken_time = current_time

    annotated_frame = results[0].plot()

    # 3 bölge çizgileri
    cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, height), (255, 0, 0), 2)
    cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, height), (255, 0, 0), 2)

    cv2.imshow('Gorme Engelli Destek Sistemi', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
