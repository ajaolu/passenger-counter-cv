import os
import sys
import cv2
import datetime
import imutils
import numpy as np
import pandas as pd
from centroidtracker import CentroidTracker

# Define paths for the pre-trained model
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Set model to use OpenCV backend and CPU target
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

# Function to handle non-max suppression
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


# Function to update Excel with new data
def update_excel(lpc_count, opc_count, fps, northing, easting, address):
    # 1. Determine where the .exe or .py file is currently located
    if getattr(sys, "frozen", False):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Combine that directory with your filename
    file_name = os.path.join(current_dir, "passenger_data.xlsx")

    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        # Create a new DataFrame and add the header
        df = pd.DataFrame(
            columns=["Coordinates", "Address", "Timestamp", "Current Passenger Count", "Total Passenger Count", "FPS"])
        # Add Northing and Easting, and address to the first row
        header_row = pd.DataFrame({"Coordinates": f"Northing: {northing}, Easting: {easting}", "Address": address},
                                  index=[0])
        df = pd.concat([header_row, df], ignore_index=True)

    new_row = {
        "Timestamp": datetime.datetime.now(),
        "Current Passenger Count": lpc_count,
        "Total Passenger Count": opc_count,
        "FPS": fps
    }

    # Python 3.10+ / Pandas Note: .append() is deprecated. Using pd.concat instead.
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_excel(file_name, index=False)

def main():
    # Predefined coordinates and address
    northing = 718221.225  # Input Northing value here
    easting = 540095.951   # Input Easting value here
    address = "Stadium Bus Stop, Surulere, Lagos."  # Input address here

    # Open video stream or webcam
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []

    last_lpc_count = -1  # Initialize to a value that won't match

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        total_frames += 1
        (H, W) = frame.shape[:2]

        # Display the predefined coordinates and address
        coords_text = f"Northing: {northing}m, Easting: {easting}m"
        address_text = f"Address: {address}"
        cv2.putText(frame, coords_text, (4, 29), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.putText(frame, address_text, (4, 59), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.putText(frame, coords_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 139), 1)
        cv2.putText(frame, address_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 139), 1)

        # Pre-process the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(person_box.astype("int"))

        rects = non_max_suppression_fast(np.array(rects), 0.3)
        objects = tracker.update(rects)

        # Display bounding boxes and object IDs
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 204, 204), 2)
            text = f"ID: {objectId}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 204, 204), 1)
            if objectId not in object_id_list:
                object_id_list.append(objectId)

        # Calculate and display FPS
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        fps = total_frames / time_diff.seconds if time_diff.seconds > 0 else 0.0
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (4, 149), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.putText(frame, fps_text, (5, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 139), 1)

        # Passenger count display
        lpc_count = len(objects)
        opc_count = len(object_id_list)
        lpc_txt = f"LIVE DETECTIONS: {lpc_count}"
        opc_txt = f"TOTAL INBOUND: {opc_count}"
        cv2.putText(frame, lpc_txt, (4, 89), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.putText(frame, opc_txt, (4, 119), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.putText(frame, lpc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 139), 1)
        cv2.putText(frame, opc_txt, (5, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 139), 1)

        # Check if the current passenger count has changed
        if lpc_count != last_lpc_count:
            # Update Excel sheet
            update_excel(lpc_count, opc_count, fps, northing, easting, address)
            last_lpc_count = lpc_count  # Update the last count to the current count

        # Display video
        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()