from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Tải mô hình phát hiện đối tượng YOLO từ tệp chỉ định. Lớp YOLO xử lý việc tải trọng lượng và cấu hình mô hình.
model = YOLO("C:/Users/admin/.vscode/python/XLA/trafficSignRecognition/bestFinal.pt") 

# Lớp chứa các đối tượng.
classNames = ['Speed limit (50km/h)', 'Speed limit (70km/h)', 'No car', 'No stopping or parking', 'No entry']

while True:
    success, img = cap.read()
    # Phát hiện đối tượng trên ảnh đã chụp bằng mô hình YOLO đã tải. Đối số stream=True chỉ ra rằng mô hình đang 
    # được sử dụng theo kiểu phát trực tuyến.
    results = model(img, stream=True)
    for r in results:
    # trích xuất các hộp giới hạn cho các đối tượng được phát hiện.
        boxes = r.boxes
        for box in boxes:
            # tọa độ của hộp giới hạn từ đối tượng hiện tại
            x1, y1, x2, y2 = box.xyxy[0] 
	        # chuyển tọa độ từ dấu phẩy động sang số nguyên
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
            # vẽ hình bao quanh đối tượng với các tọa độ đã xác định
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # độ chính xác của đối tượng được nhận dạng
            confidence = math.ceil((box.conf[0] * 100))
            # trích xuất chỉ mục lớp từ đối tượng hộp giới hạn hiện tại, chuyển đổi chỉ mục lớp thành số nguyên 
            # để xử lý tiếp.
            cls = int(box.cls[0])
	        # chuỗi chứa tên đối tượng và độ chính xác khi nhận dạng
            class_text = f"{classNames[cls]} ({confidence}%)"  
            # tọa độ thể hiện vị trí hiển thị của chuỗi class_text trên màn hình
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
	        # độ dày của văn bản
            thickness = 2
	        # hiển thị class_text lên màn hình với các thông số được thiết lập như trên
            if confidence > 70:
                cv2.putText(img, class_text, org, font, fontScale, color, thickness)
            else:
                class_text = "Unknown"
                cv2.putText(img, class_text, org, font, fontScale, color, thickness)
    
    cv2.imshow('TRAFFIC SIGN RECOGNITION', img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
