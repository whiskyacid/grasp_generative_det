import sys
sys.path.append('/home/jian/CV/ur_grcnn_grasp')
from RealsenseCamera.camera import RealsenseCamera
import cv2
import torch
import numpy as np
from inference.processing import result_process,result_plot,detect_grasps
import matplotlib.pyplot as plt
from Grasper.Grasp import Grasp
device = torch.device("cuda") # "cuda" or "cpu"
Camera =RealsenseCamera()
Camera.camera_connect()
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolov5_model.to(device)
model = torch.load('/home/jian/CV/ur_grcnn_grasp/inference/weights/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98')
while True:
    depth_frame,color_frame = Camera.get_align_frames()
    depth_image = Camera.get_depth_image(depth_frame)
    color_image_origenal,color_image = Camera.get_color_image(color_frame)
    img2torch = Camera.numpy2torch(color_image,depth_image)
    with torch.no_grad():
        predict = model.predict(img2torch.to(device))
        results = yolov5_model(color_image_origenal)
    q_img, ang_img, width_img = result_process(predict['pos'], predict['cos'], predict['sin'], predict['width'])
    boxs= results.pandas().xyxy[0].values
    grasps = []
    for box in boxs:
        q_box = q_img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        max_index_q = np.unravel_index(np.argmax(q_box), q_box.shape)  #(y,x)  二维数组索引顺序：先行，后列
        max_index_q_gloab = (max_index_q[0]+int(box[1]),max_index_q[1]+int(box[0]))
        grasp_center = (max_index_q_gloab[1],max_index_q_gloab[0])
        grasp_angle = ang_img[max_index_q_gloab[0],max_index_q_gloab[1]]
        grasp_width = width_img[max_index_q_gloab[0],max_index_q_gloab[1]]
        g = Grasp(grasp_center,grasp_angle,grasp_width)
        grasps.append(g)

    for g in grasps:
        g.GraspRectangle(color_image_origenal)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break