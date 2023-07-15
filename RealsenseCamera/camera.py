import pyrealsense2 as rs
import numpy as np
import torch
import cv2
class RealsenseCamera:

    def __init__(self,width=640,height=480,output=224):
        self.width = width
        self.height = height
        self.output = output
        self.pipeline = None
        self.scale = None
        self.align = rs.align(rs.stream.color)

        # self.spatial = rs.spatial_filter()
        # self.spatial.set_option(rs.option.filter_magnitude,3)
        # self.spatial.set_option(rs.option.filter_smooth_alpha,0.5) # 0.25-1
        # self.spatial.set_option(rs.option.filter_smooth_delta,25) # 0-50
        # self.hole_filling = rs.hole_filling_filter()
        # self.hole_filling.set_option(rs.option.holes_fill, 2)

    def camera_connect(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        cfg = self.pipeline.start(config)
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_align_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame,color_frame

    def get_depth_image(self,depth_frame):
        # frame = self.spatial.process(depth_frame)
        # frame = self.hole_filling.process(frame)
        depth_image = np.asanyarray(depth_frame.get_data(),dtype=np.float32)
        depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        depth_image = depth_image[(self.height - self.output) // 2:(self.height + self.output) // 2 ,
                                  (self.width - self.output) // 2:(self.width + self.output) // 2, 
                                  : ]
        depth_image = depth_image / 255.0
        depth_image -= depth_image.mean()
        depth_image = np.array(depth_image)
        depth_image = depth_image.transpose((2,0,1)) #（1，H，W）
        return depth_image

    def get_color_image(self,colo_frame):
        color_image_origenal = np.asanyarray(colo_frame.get_data())
        color_image_origenal = color_image_origenal[(self.height - self.output) // 2:(self.height + self.output) // 2 ,
                                  (self.width - self.output) // 2:(self.width + self.output) // 2, 
                                  : ]
        color_image = color_image_origenal.astype(np.float32)
        # color_image = color_image[(self.height - self.output) // 2:(self.height + self.output) // 2 ,
        #                           (self.width - self.output) // 2:(self.width + self.output) // 2, 
        #                           : ]
        color_image = color_image / 255.0
        color_image = color_image - color_image.mean()
        color_image = np.array(color_image)
        color_image = color_image.transpose((2,0,1))
        return color_image_origenal,color_image

    def numpy2torch(self,color_image,depth_image):
        img2torch = torch.from_numpy(
                    np.concatenate(
                    (np.expand_dims(depth_image, 0),  # (1,1,224,224)
                        np.expand_dims(color_image, 0)),  # (1,3,224,224)
                    1                                   # (1,4,224,224)
                )   
            )
        return img2torch
    
