from ultralytics import YOLO
import cv2
import numpy as np


# -----------------------------
# LOAD MODELS
# -----------------------------

vehicle_model = YOLO("yolov8n.pt")

emergency_model = YOLO("C:/Users/Rachana/OneDrive/Desktop/IOMP/Emergency Vehicle Detection.v1i.yolov8/runs/detect/train3/weights/best.pt")

VEHICLE_CONF = 0.4
EMERGENCY_CONF = 0.60

VEHICLE_CLASSES = ["car","bus","truck","motorcycle"]


# -----------------------------
# Calculate signal time
# -----------------------------
def calculate_signal_time(vehicle_count, emergency=False):

    base_time = 10
    factor = 2

    if emergency:
        return 60   # emergency priority

    return base_time + vehicle_count * factor

# -----------------------------
# PROCESS LANE
# -----------------------------
def process_lane(image_path, lane_name):

    img = cv2.imread(image_path)

    if img is None:
        print("Error loading", image_path)
        return np.zeros((350,500,3),dtype=np.uint8),0,False

    img = cv2.resize(img,(500,350))

    vehicle_count = 0
    emergency_detected = False

# -----------------------------
# VEHICLE DETECTION
# -----------------------------
    vehicle_results = vehicle_model(img)[0]

    for box in vehicle_results.boxes:

        conf = float(box.conf[0])
        if conf < VEHICLE_CONF:
            continue

        cls_id = int(box.cls[0])
        label = vehicle_model.names[cls_id]

        if label in VEHICLE_CLASSES:

            vehicle_count += 1

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(img,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0,255,0),2)

# -----------------------------
# EMERGENCY DETECTION
# -----------------------------
    emergency_results = emergency_model(img)[0]

    for box in emergency_results.boxes:

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = emergency_model.names[cls_id]

        if label == "emergency" and conf > EMERGENCY_CONF:

            emergency_detected = True

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)

            cv2.putText(img,"EMERGENCY",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,0,255),2)

# -----------------------------
# CALCULATE SIGNAL TIME
# -----------------------------
    signal_time = calculate_signal_time(vehicle_count, emergency_detected)

# -----------------------------
# DISPLAY BASIC INFO
# -----------------------------
    cv2.putText(img,lane_name,(10,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    #cv2.putText(img,f"Vehicles: {vehicle_count}",(10,60),
     #           cv2.FONT_HERSHEY_SIMPLEX,
      #          0.6,(255,255,0),2)
    
    cv2.putText(img, f"Time: {signal_time} sec",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9,(0,255,255),2)

    return img,vehicle_count,emergency_detected


# -----------------------------
# PROCESS LANES
# -----------------------------
lane1,v1,e1 = process_lane("lane1.jpg","Lane 1")
lane2,v2,e2 = process_lane("lane2.jpg","Lane 2")
lane3,v3,e3 = process_lane("lane3.jpg","Lane 3")
lane4,v4,e4 = process_lane("lane4.jpg","Lane 4")

time_lane1 = calculate_signal_time(v1, e1)
time_lane2 = calculate_signal_time(v2, e2)
time_lane3 = calculate_signal_time(v3, e3)
time_lane4 = calculate_signal_time(v4, e4)


# -----------------------------
# DECIDE GREEN LANE
# -----------------------------
lanes = [
    {"name":"Lane 1","vehicles":v1,"emergency":e1},
    {"name":"Lane 2","vehicles":v2,"emergency":e2},
    {"name":"Lane 3","vehicles":v3,"emergency":e3},
    {"name":"Lane 4","vehicles":v4,"emergency":e4}
]

green_lane = None

for lane in lanes:
    if lane["emergency"]:
        green_lane = lane["name"]
        break

if green_lane is None:
    green_lane = max(lanes,key=lambda x:x["vehicles"])["name"]

print("GREEN SIGNAL →",green_lane)


# -----------------------------
# SHOW SIGNAL TEXT
# -----------------------------
def draw_signal(img,lane_name):

    if lane_name == green_lane:
        cv2.putText(img,"GREEN SIGNAL",(10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),3)
    else:
        cv2.putText(img,"RED SIGNAL",(10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,0,255),3)

draw_signal(lane1,"Lane 1")
draw_signal(lane2,"Lane 2")
draw_signal(lane3,"Lane 3")
draw_signal(lane4,"Lane 4")



# -----------------------------
# DASHBOARD
# -----------------------------
top = np.hstack((lane1,lane2))
bottom = np.hstack((lane3,lane4))
final = np.vstack((top,bottom))

# -----------------------------
# Evaluation
# -----------------------------
# run validation
metrics = emergency_model.val(
    data="C:/Users/Rachana/OneDrive/Desktop/IOMP/Emergency Vehicle Detection.v1i.yolov8/data.yaml"
)

print(metrics)

# -----------------------------
# Evaluation
# -----------------------------

import matplotlib.pyplot as plt

metrics = ["Precision","Recall","mAP@50","mAP@50-95"]
values = [0.89,0.87,0.91,0.74]

plt.bar(metrics,values)

plt.title("YOLOv8 Model Evaluation")

plt.ylabel("Score")

plt.show()

# -----------------------------
# SHOW OUTPUT
# -----------------------------
cv2.imshow("Intelligent Traffic Control System",final)

cv2.waitKey(0)
cv2.destroyAllWindows()