from ultralytics import YOLO

model = YOLO('D:/HUST/dev/py/Digit_Reg/model/best_train35.pt')

results = model('D:/HUST/dev/py/Digit_Reg/fileZip/test/images/27832629_1.jpeg')
        

boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()
boxes_sorted = sorted(zip(boxes, classes, confidences), key=lambda x: x[0][0], reverse=False)
detected_number = ""
for box, cls, conf in boxes_sorted:
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        detected_number += name
        
print(detected_number)
for result in results:
    result.show()