import numpy as np
import cv2
class Grasp:
    def __init__(self,grasp_center,grasp_angle,grasp_width):
        self.grasp_center = grasp_center
        self.grasp_angle = grasp_angle
        self.grasp_width = grasp_width

    def GraspRectangle(self,image):
        print("grasp_angle = ",self.grasp_angle)
        print("grasp_center = ",self.grasp_center)
        cos = np.cos(-self.grasp_angle)
        sin = np.sin(-self.grasp_angle)
        cx = self.grasp_center[0]
        cy = self.grasp_center[1]
        half_width = self.grasp_width / 2
        half_height = 10
        print("grasp_width = ",self.grasp_width)

        top_left = (int(cx - half_width * cos - half_height * sin), int(cy - half_width * sin + half_height * cos))
        top_right = (int(cx + half_width * cos - half_height * sin), int(cy + half_width * sin + half_height * cos))
        bottom_left = (int(cx - half_width * cos + half_height * sin), int(cy - half_width * sin - half_height * cos))
        bottom_right = (int(cx + half_width * cos + half_height * sin), int(cy + half_width * sin - half_height * cos))

        # top_left = (int(cx - half_width * cos - half_width * sin),int(cy - half_width * sin + half_width * cos))
        # top_right = (int(cx + half_width * cos - half_width * sin),
        #             int(cy + half_width * sin + half_width * cos))
        # bottom_left = (int(cx - half_width * cos + half_width * sin),
        #             int(cy - half_width * sin - half_width * cos))
        # bottom_right = (int(cx + half_width * cos + half_width * sin),
        #                 int(cy + half_width * sin - half_width * cos))
        

    #     return top_left,top_right,bottom_right,bottom_left

    # def draw_rectangle(self,image, top_left, top_right, bottom_right, bottom_left):
    #     # 绘制矩形的四条边
        cv2.line(image, top_left, top_right, (0, 255, 0), thickness=1)     # 绘制上边
        cv2.line(image, top_right, bottom_right, (0, 0, 255), thickness=1) # 绘制右边
        cv2.line(image, bottom_right, bottom_left, (0, 255, 0), thickness=1) # 绘制下边
        cv2.line(image, bottom_left, top_left, (0, 0, 255), thickness=1)  # 绘制左边
        cv2.imshow("grasp",image)