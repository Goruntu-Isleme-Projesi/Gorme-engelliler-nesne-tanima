from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# Son sesli uyarı zamanı
last_spoken = 0

# Modeli yükle
model = YOLO('yolov8n.pt')

# Kamerayı aç
url = "http://192.168.1.17:4747/video"
cap = cv2.VideoCapture(url)

# Sesli uyarı için bir kilit (Lock) sistemi kuruyoruz
speak_lock = threading.Lock()

def speak(text):
    def run():
        with speak_lock:  # aynı anda sadece bir seslendirme olsun
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    threading.Thread(target=run).start()

# Nesne türüne göre gerçek genişlikler (metre cinsinden)
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

# Kamera sabit odak uzunluğu
focal_length = 615  # kalibre edilmiş

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, _ = frame.shape

    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False, device=0)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        classes = boxes.cls.cpu().numpy().astype(int)

        current_time = time.time()

        if current_time - last_spoken > 4:
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

                # Gerçek genişliği al
                real_width = real_widths.get(label, 0.5)  # eğer tanımlı değilse 0.5 metre kabul et

                # Mesafe hesapla
                pixel_width = x2 - x1
                if pixel_width > 0:
                    distance = (real_width * focal_length) / pixel_width
                else:
                    distance = -1

                # Mesaj oluştur
                if distance != -1:
                    print(f"{label} detected in the {position} region, approximately {distance:.1f} meters away")
                    text = f"{label} detected in the {position} region, approximately {distance:.1f} meters away"
                else:
                    print(f"{label} detected in the {position} region")
                    text = f"{label} detected in the {position} region"

                # Sesli uyarıyı gönder
                speak(text)

                last_spoken = current_time
                break

    annotated_frame = results[0].plot()

    # 9 parçaya bölme çizgileri
    cv2.line(annotated_frame, (width // 3, 0), (width // 3, height), (255, 0, 0), 2)
    cv2.line(annotated_frame, (2 * width // 3, 0), (2 * width // 3, height), (255, 0, 0), 2)
    cv2.line(annotated_frame, (0, height // 3), (width, height // 3), (255, 0, 0), 2)
    cv2.line(annotated_frame, (0, 2 * height // 3), (width, 2 * height // 3), (255, 0, 0), 2)

    cv2.imshow('YOLO Detection with Location and Distance', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
