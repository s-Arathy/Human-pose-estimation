import cv2 as cv
import numpy as np
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog


def estimate_pose(image_data, confidence_threshold=0.2):
    
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }
    
    POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
    ]
    
    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    
 
    if isinstance(image_data, bytes):
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
    elif isinstance(image_data, np.ndarray):
        frame = image_data
    else:
        raise ValueError("Unsupported image format")
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    
    pose_data = {
        'keypoints': {},
        'connections': [],
        'image_size': {'width': frameWidth, 'height': frameHeight}
    }
    
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        
        if conf > confidence_threshold:
            pose_data['keypoints'][list(BODY_PARTS.keys())[i]] = {
                'position': {'x': int(x), 'y': int(y)},
                'confidence': float(conf)
            }
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        if points[idFrom] and points[idTo]:
            pose_data['connections'].append({
                'from': partFrom,
                'to': partTo,
                'from_pos': {'x': points[idFrom][0], 'y': points[idFrom][1]},
                'to_pos': {'x': points[idTo][0], 'y': points[idTo][1]}
            })
    
    return pose_data

def visualize_pose(image_data, pose_data):
    
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
    elif isinstance(image_data, np.ndarray):
        frame = image_data.copy()
    else:
        raise ValueError("Unsupported image format")
    
    
    for connection in pose_data['connections']:
        start_point = (connection['from_pos']['x'], connection['from_pos']['y'])
        end_point = (connection['to_pos']['x'], connection['to_pos']['y'])
        cv.line(frame, start_point, end_point, (0, 255, 0), 3)
    
    
    for keypoint, data in pose_data['keypoints'].items():
        point = (data['position']['x'], data['position']['y'])
        cv.ellipse(frame, point, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    
    return frame

root = tk.Tk()
root.withdraw()  
    
file_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
        ("All files", "*.*")
    ]
)
    
    
print(f"Processing image: {file_path}")
    
    
with open(file_path, 'rb') as f:
    image_data = f.read()

pose_data = estimate_pose(image_data)
result_image = visualize_pose(image_data, pose_data)
cv.imshow('Pose Estimation', result_image)
cv.waitKey(0)
cv.destroyAllWindows()
for keypoint, data in pose_data['keypoints'].items():
    print(f"{keypoint}: position={data['position']}, confidence={data['confidence']:.2f}")

