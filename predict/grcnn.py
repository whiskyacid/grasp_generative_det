import sys
sys.path.append('/home/jian/CV/ur_grcnn_grasp')
from RealsenseCamera.camera import RealsenseCamera
import cv2
import torch
from inference.processing import result_process,result_plot,detect_grasps
import matplotlib.pyplot as plt
Camera =RealsenseCamera()
Camera.camera_connect()
device = torch.device("cuda") # "cuda" or "cpu"
model = torch.load('/home/jian/CV/ur_grcnn_grasp/inference/weights/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98')
fig = plt.figure(figsize=(12, 4))
while True:
    depth_frame,color_frame = Camera.get_align_frames()
    depth_image = Camera.get_depth_image(depth_frame)
    color_image_origenal,color_image = Camera.get_color_image(color_frame)
    img2torch = Camera.numpy2torch(color_image,depth_image)
    with torch.no_grad():
        predict = model.predict(img2torch.to(device))

    q_img, ang_img, width_img = result_process(predict['pos'], predict['cos'], predict['sin'], predict['width'])
    grasps = detect_grasps(q_img, ang_img, width_img,nu_grasps=1)
    for g in grasps:
        g.GraspRectangle(color_image_origenal)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

    result_plot(fig,q_img, ang_img, width_img,depth_image)