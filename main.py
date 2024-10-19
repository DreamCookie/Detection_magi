import cv2
import numpy as np

# === Шаг 1: Загрузка классов и модели YOLOv3 ===
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# Использование GPU, если доступно (раскомментируйте при необходимости)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# === Шаг 2: Инициализация видеопотока ===
# Захват видеопотока с веб-камеры (индекс 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

# === Шаг 3: Инициализация трекера ===
trackers = cv2.MultiTracker_create()
init_once = False

# === Шаг 4: Определение зоны интереса (ROI) ===
roi_start = (100, 100)
roi_end = (500, 500)

# === Основной цикл обработки кадров ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получение размеров кадра
    height, width, _ = frame.shape

    # Рисование зоны интереса на кадре
    cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)

    if not init_once:
        # === Шаг 5: Обнаружение объектов с помощью YOLOv3 ===

        # Преобразование изображения в blob
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Получение имен выходных слоев
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        # Списки для обнаруженных объектов
        boxes = []
        confidences = []
        class_ids = []

        # Обработка выходных данных
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Фильтрация по классу 'person' и порогу доверия
                if confidence > 0.5 and class_id == 0:
                    # Координаты обнаруженного объекта
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Вычисление координат верхнего левого угла
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Подавление нефрагментированных объектов
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Инициализация трекеров для обнаруженных объектов
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                # Ограничение координат рамки в пределах кадра
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                bbox = (x, y, w, h)
                # Использование трекера CSRT
                tracker = cv2.TrackerCSRT_create()
                trackers.add(tracker, frame, bbox)
            init_once = True
    else:
        # === Шаг 6: Обновление трекеров ===

        success, boxes = trackers.update(frame)

        for i, newbox in enumerate(boxes):
            x, y, w, h = map(int, newbox)
            # Рисование рамки вокруг отслеживаемого объекта
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Вычисление центра объекта
            cx = x + w // 2
            cy = y + h // 2

            # Проверка, находится ли объект внутри зоны интереса
            if roi_start[0] < cx < roi_end[0] and roi_start[1] < cy < roi_end[1]:
                cv2.putText(frame, 'Alert: Person in Zone!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # === Шаг 7: Отображение кадра ===
    cv2.imshow('Real-Time Person Tracking', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Выход из программы при нажатии 'q'
        break
    elif key == ord('r'):
        # Сброс трекеров при нажатии 'r'
        trackers = cv2.MultiTracker_create()
        init_once = False

# === Завершение ===
cap.release()
cv2.destroyAllWindows()