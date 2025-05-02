from ultralytics import YOLO
import cv2
import pyttsx3
import time
import threading

# Son sesli uyarı zamanı
last_spoken = 0

# Son algılanan nesne bilgileri
last_label = None
last_position = None
last_distance = None

# Modeli yükle
model = YOLO('yolov8n.pt')

# Kamerayı aç
url = "http://192.168.1.17:4747/video"
cap = cv2.VideoCapture(1)

# Sesli uyarı için bir kilit (Lock) sistemi kuruyoruz
speak_lock = threading.Lock()

def speak(text):
    def run():
        with speak_lock:
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
focal_length = 615

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, _ = frame.shape

    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False, device='cpu')

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = results[0].names
        classes = boxes.cls.cpu().numpy().astype(int)

        current_time = time.time()

        if current_time - last_spoken > 4:
            closest_obj = None
            closest_distance = float('inf')
            closest_info = None

            for cls_idx, box in zip(classes, boxes.xyxy):
                label = names[cls_idx]
                x1, y1, x2, y2 = box.cpu().numpy()

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

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

                real_width = real_widths.get(label, 0.5)
                pixel_width = x2 - x1
                if pixel_width > 0:
                    distance = (real_width * focal_length) / pixel_width
                else:
                    distance = -1

                if distance != -1 and distance < closest_distance:
                    closest_distance = distance
                    closest_obj = label
                    closest_info = (position, distance)

            if closest_obj:
                position, distance = closest_info

                # Eğer zaten aynı nesne ve mesafedeysek tekrar konuşma
                if (closest_obj == last_label and
                    position == last_position and
                    last_distance is not None and
                    abs(distance - last_distance) < 0.3):
                    pass
                else:
                    print(f"{closest_obj} detected in the {position} region, approximately {distance:.1f} meters away")
                    text = f"{closest_obj} detected in the {position} region, approximately {distance:.1f} meters away"
                    speak(text)

                    last_label = closest_obj
                    last_position = position
                    last_distance = distance
                    last_spoken = current_time

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
