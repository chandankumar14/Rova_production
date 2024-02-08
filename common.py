import numpy as np
from io import BytesIO
from PIL import Image
import cv2
config_file ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
class_labels=['person', 'bicycle', 'car', 'motorcycle', 
'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 
'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 
'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup'
,'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 
'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 
'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def classification(image):
    preObj =[]
    status =False
    model = cv2.dnn_DetectionModel(frozen_model,config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5) 
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.7)
    if len(ClassIndex)>0:
        status =True
        for index in ClassIndex:
            value =index-1
            preObj.append(class_labels[value])
           
    return {
        'status':status,
        'ObjectList':preObj,
        'Message':'image is contaning different object, please use correct image ',
        'StatusCode':401
    }

        
    

    
