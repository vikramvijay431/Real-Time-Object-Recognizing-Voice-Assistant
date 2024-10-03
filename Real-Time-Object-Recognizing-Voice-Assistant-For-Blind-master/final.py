import cv2
import numpy as np

# Assuming you have downloaded and configured YOLO (e.g., darknet)
net = cv2.dnn.readNetFromDarknet(r"C:\Users\ANANTH\.idlerc\Downloads\Real-Time-Object-Recognizing-Voice-Assistant-For-Blind-master\yolov3.cfg", r"C:\Users\ANANTH\.idlerc\Downloads\Real-Time-Object-Recognizing-Voice-Assistant-For-Blind-master\yolov3.weights")
names =open( r"C:\Users\ANANTH\.idlerc\Downloads\Real-Time-Object-Recognizing-Voice-Assistant-For-Blind-master\coco.names")  # Replace with your class names

def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getLayerNames()
    output_layers_names = [output_layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers_names)
    return outputs

def inpaint_object(frame, selected_object_box, background):
    """
    Inpaints the selected object using the provided background image.

    Args:
        frame: The original frame.
        selected_object_box: A tuple (startX, startY, endX, endY) defining the object's bounding box.
        background: The background image used for inpainting.

    Returns:
        The inpainted frame.
    """

    frame_copy = frame.copy()
    cv2.inpaint(frame_copy, selected_object_box, background, 3, cv2.INPAINT_TELEA)
    return frame_copy

selected_object = None
selected_object_box = None
background = None  # Initialize background image for inpainting

def on_mouse_click(event, x, y, flags, param):
    global selected_object, selected_object_box
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click falls within any detected object bounding box
        for outputs in outputs:
            for detection in outputs:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))
                    endX = startX + width
                    endY = startY + height

                    if startX <= x <= endX and startY <= y <= endY:
                        selected_object = class_id
                        selected_object_box = (startX, startY, endX, endY)
                        # Capture background once on first object selection
                        if background is None:
                            background = frame.copy()
                        break  # Stop iterating after finding the clicked object

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    outputs = detect_objects(frame)

    cv2.setMouseCallback("Frame", on_mouse_click, outputs)

    # Make selected object invisible (if any)
    if selected_object is not None:
        frame = inpaint_object(frame, selected_object, background)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
