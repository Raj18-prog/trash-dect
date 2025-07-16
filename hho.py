import cv2
import numpy as np
import torch

def preprocess_image(image_path):
    image = cv2.imread('t2.mp4')
    image = cv2.resize(image, (640, 640))  # Resize image to 640x640 for YOLOv5
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model
    return model

def detect_plastic_trash(model, image):
    results = model(image)  # Pass image to YOLO model for inference
    results.render()  # Render the bounding boxes and labels on the image
    return results

def show_results(image, results):
    cv2.imshow("Detected Plastic Trash", image)  # Display image with detected objects
    cv2.waitKey(0)
    cv2.destroyAllWindows()
