from ultralytics import YOLO
import tensorflow as tf
import cv2

model = YOLO('D:/HUST/dev/py/Digit_Reg/model/1k_150epochs_best.pt')

cap = cv2.VideoCapture(0)

num_classes = 10  # Number of classes
detected_objects = []
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    results = model.predict(frame)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    boxes_sorted = sorted(zip(boxes, classes, confidences), key=lambda x: x[0][1], reverse=True)
    detected_number = ""

    # Iterate through the results
    for box, cls, conf in boxes_sorted:
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]

        if conf > 0.25:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{name}: {confidence:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,f"Predicted : {detected_number}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2, cv2.LINE_AA,)
            detected_number += name
      
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
