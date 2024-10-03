import cv2 as cv 
import numpy as np
import pyttsx3

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open(r"C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet(r'C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\yolov4-tiny.weights', r'C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Initialize the TTS engine
engine = pyttsx3.init()

# object detector function /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each object based on its class id 
        color = COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid], score)  # Directly use classid
        
        # draw rectangle and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where to draw text (distance)
        data_list.append([class_names[classid], box[2], (box[0], box[1]-2), (box[0] + box[2] // 2, box[1] + box[3] // 2)])
    # returning a list containing the object data 
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# reading the reference image from dir 
ref_person = cv.imread(r'C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\ReferenceImages\image14.png')
ref_mobile = cv.imread(r'C:\Users\ANANTH\.idlerc\Downloads\Yolov4-Detector-and-Distance-Estimator-master\ReferenceImages\image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels: {person_width_in_rf}, mobile width in pixels: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        class_name, object_width, text_position, object_center = d
        distance = distance_finder(focal_person if class_name == 'person' else focal_mobile, 
                                   PERSON_WIDTH if class_name == 'person' else MOBILE_WIDTH, 
                                   object_width)
        x, y = text_position
        # Announce the detected object and distance
        engine.say(f"A {class_name} is {round(distance, 2)} inches away.")
        
        # Determine direction
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        if object_center[0] < frame_center[0]:
            direction = "left"
        else:
            direction = "right"
        
        engine.say(f"It is located to the {direction} of the screen.")
        
        # Draw rectangle and text on frame
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1 )
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        
    cv.imshow('frame', frame)
    
    # Start the speech engine
    engine.runAndWait()
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
        
cv.destroyAllWindows()
cap.release()
