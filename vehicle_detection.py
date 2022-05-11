import cv2 
import csv
import numpy as np
import timeit
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Model, weight, and classfile
modelconfig = "yolov4-tiny.cfg"
modelweight = "yolov4-tiny.weights"
classfile = "coco.names"
classnames = open(classfile).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(modelconfig, modelweight)

# Class index for our required detection classes
required_class_index = [2, 3, 5, 7]
detected_classnames = []

# Load video
cap = cv2.VideoCapture("vid1.mp4")
input_size = 320

# Detection confidence threshold
confthreshold = 0.2
nmsthreshold = 0.2

left_x1 = 430 ; left_x2 = 510
straight_x1 = 330 ; straight_x2 = 410
right_x1 = 230  ; right_x2 = 310
line = 255
error = 15

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_thickness = 2
font_color = (0, 0 , 255)
colors = np.random.randint(0, 255, size=(len(classnames), 3), dtype='uint8')

total_time = []

def find_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detected_list = []
right_list = [0, 0, 0, 0]
straight_list = [0, 0, 0, 0]
left_list = [0, 0, 0, 0]

def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center

    if (iy > (line - error)) and (iy < line):
        if id not in detected_list:
            detected_list.append(id)
            if (ix > left_x1) and (ix < left_x2):
                left_list[index] = left_list[index]+1
            if (ix > straight_x1) and (ix < straight_x2):
                straight_list[index] = straight_list[index]+1
            if (ix > right_x1) and (ix < right_x2):
                right_list[index] = right_list[index]+1
            total_time.append(detect_time)

    cv2.circle(img, center, 2, (0, 0, 255), -1)

def postprocess(outputs, img):
    global detect_time
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confthreshold:
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))
                    detect_time = timeit.default_timer() - start_time

    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confthreshold, nmsthreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        color = [int(c) for c in colors[classIds[i]]]
        name = classnames[classIds[i]]
        detected_classnames.append(name)
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%', (x, y-10), font, 0.5, color, 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])
    
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

def realTime():
    while True:
        global start_time 
        start_time = timeit.default_timer()
        success, img = cap.read()
        img = cv2.resize(img, (0,0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersnames = net.getLayerNames()
        outputnames = [(layersnames[i - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputnames)
        postprocess(outputs, img)

        cv2.line(img, (left_x1, line), (left_x2, line), (255, 0, 0), 3)
        cv2.line(img, (straight_x1, line), (straight_x2, line), (0, 255, 0), 3)
        cv2.line(img, (right_x1, line), (right_x2, line), (0, 0, 255), 3)

        cv2.putText(img, "Right", (110, 20), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Straight", (160, 20), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Left", (230, 20), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:         "+str(right_list[0])+"      "+str(straight_list[0])+"      "+str(left_list[0]), (20, 40), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:   "+str(right_list[1])+"      "+str(straight_list[1])+"      "+str(left_list[1]), (20, 60), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:         "+str(right_list[2])+"      "+str(straight_list[2])+"      "+str(left_list[2]), (20, 80), font, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:       "+str(right_list[3])+"      "+str(straight_list[3])+"      "+str(left_list[3]), (20, 100), font, font_size, font_color, font_thickness)

        cv2.imshow('Output', img)
        print(detected_list)
        print(total_time)
        if cv2.waitKey(1) == ord('q'):
            break

    with open("data1.csv", "w") as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'Car', 'Motorbike', 'Bus', 'Truck'])
        right_list.insert(0, "Right")
        straight_list.insert(0, "Straight")
        left_list.insert(0, "Left")
        cwriter.writerow(right_list)
        cwriter.writerow(straight_list)
        cwriter.writerow(left_list)
    f1.close()
    print("Data saved at 'data1.csv'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realTime()