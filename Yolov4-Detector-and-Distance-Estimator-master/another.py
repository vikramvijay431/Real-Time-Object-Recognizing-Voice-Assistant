import cv2
import numpy as np
from imutils.video import FPS

# Load COCO class names (replace with your path)
with open(r"C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\coco.names", "r") as f:
    coco_names = [line.strip() for line in f.readlines()]

# Convert COCO class names to dictionary (assuming class IDs start from 0 and match list indices)
coco_names_dict = {i: name for i, name in enumerate(coco_names)}

# Load pre-trained object detection model (replace with your model paths)
weightsPath = r"C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\frozen_inference_graph (1).pb"  # Example path for TensorFlow model
configPath = r"C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\mask_rcnn_inception_v2_coco_2018_01_28 (1).pbtxt"  # Example path for Mask R-CNN config file

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

global clicked_object 
clicked_object = None
# ... rest of your code ...

# Define function to identify the clicked object
def identify_clicked_object(detected_objects, click_x, click_y):
    clicked_object = None
    min_distance = float("inf")
    for obj in detected_objects:
        x, y, w, h = obj["box"]
        center_x = (x + x + w) // 2
        center_y = (y + y + h) // 2
        distance = np.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            clicked_object = obj
    return clicked_object

# Mouse click event handler
def click_event(event, x, y, flags, param):
    global clicked_object, invisibility_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        detected_objects, frame = param
        clicked_object = identify_clicked_object(detected_objects, x, y)

        # Create a copy for invisibility mask (if object clicked)
        if clicked_object is not None:
            invisibility_frame = frame.copy()

# Function to draw polygon (adapted for clarity and efficiency)
def draw_polygon(img, polygon, color=(0, 255, 0), thickness=2):
    if polygon:
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [points], True, color, thickness)

# Initialize variables
webcam = 0
expected_confidence = 0.3
threshold = 0.1
show_output = True
save_output = False  # Set to True if needed

# Capture video stream
cap = cv2.VideoCapture(webcam)

if not cap.isOpened():
    print("[ERROR] Failed to open video stream!")
    exit()

# Capture first frame for background
ret, frame = cap.read()
if not ret:
    print("[ERROR] Failed to capture frame!")
    exit()

# Create background subtractor (optional, adjust parameters if needed)
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10)

cv2.namedWindow("Detection Frame")
cv2.namedWindow("Invisibility Mask")
cv2.setMouseCallback("Detection Frame", click_event, param=(None, None))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame for detection
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    detected_objects = []
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        if confidence > expected_confidence:
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")

        # Extract polygon coordinates (assuming Mask R-CNN provides them)
        # You might need to modify this based on your model's output format
        # (e.g., access specific layers or attributes)
        if "rois" in masks:  # Check if polygon coordinates are available in the output
            rois = masks['rois'][0]  # Assuming polygon coordinates are in 'rois' key within masks
            x1, y1, x2, y2, x3, y3, x4, y4 = rois[i * 4: (i * 4) + 8]
            polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # Create polygon from coordinates
        else:
            # Fallback to bounding boxes if polygon coordinates are not available
            polygon = [(startX, startY), (startX, endY), (endX, endY), (endX, startY)]

        detected_objects.append({
            "box": (startX, startY, endX - startX, endY - startY),
            "mask": masks[i, classID],
            "class_id": classID,
            "polygon": polygon
        })

    # Create a copy for invisibility mask (if object clicked)
    invisibility_frame = None
    if clicked_object is not None:
        invisibility_frame = frame.copy()

    # Process frame for background subtraction (optional)
    fgmask = fgbg.apply(frame)

    # Handle invisibility based on clicked object
    if clicked_object is not None:
        x, y, w, h = clicked_object["box"]
        mask = cv2.resize(clicked_object["mask"], (w, h))

        # Process mask for invisibility (e.g., alpha blending)
        alpha = 0.7  # Adjust alpha value for desired transparency (0 for fully transparent, 1 for fully opaque)
        beta = 1 - alpha
        invisibility_frame[y:y + h, x:x + w] = cv2.addWeighted(invisibility_frame[y:y + h, x:x + w], alpha, mask, beta, 0)

    # Draw polygons and class names on detection frame
    for obj in detected_objects:
        x, y, w, h = obj["box"]
        class_id = obj["class_id"]
        class_name = coco_names_dict.get(class_id, "unknown")
        polygon = obj["polygon"]

        draw_polygon(frame, polygon)

        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show both frames
    cv2.imshow("Detection Frame", frame)
    if invisibility_frame is not None:
        cv2.imshow("Invisibility Mask", invisibility_frame)

    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

