import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from camera import Camera

import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from training import get_model_instance_segmentation, get_transform

import open3d as o3d


MIN_SCORE = 0.7

def get_mask_center(mask):
    "return pixel coordinates of mask center"
    mask_center = np.mean(np.argwhere(mask),axis=0)

    return int(mask_center[1]), int(mask_center[0])

def mask_vertices_colors(mask, color_image, aligned_depth_frame):
    idxes = np.argwhere(mask)
    points = []
    colors = []
    for row in idxes:
        coords = camera.deproject_pixel([row[1],row[0]],aligned_depth_frame)
        points.append(coords)
        colors.append(color_image[row[0],row[1], :])

    return np.asarray(points).astype(np.float32), np.asarray(colors).astype(np.uint8)

def get_surface_normal(mask):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mask)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    for i in range(np.asarray(pcd.normals).shape[0]):
        if pcd.normals[i][2] > 0:
            pcd.normals[i][0] = -pcd.normals[i][0]
            pcd.normals[i][1] = -pcd.normals[i][1]
            pcd.normals[i][2] = -pcd.normals[i][2]
    
    normals = np.asarray(pcd.normals)
    return np.sum(normals, axis=0) / normals.shape[0]




if __name__ == '__main__':

    camera = Camera()
    time.sleep(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load('C:/Users/anazi/Documents/digging/cross.pt', map_location=device))
    model.eval().to(device)

    eval_transform = get_transform(train=False)

    while True:
        color_frame, aligned_depth_frame = camera.get_frames()

        color_image, depth_image, vertices, vertices_color = camera.process_frames(color_frame, aligned_depth_frame)
        image = torch.as_tensor(color_image).permute(2,0,1)

        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        pred_labels = [f"cross: {score:.3f}" for label, score in zip(pred["labels"][pred['scores']>MIN_SCORE], pred["scores"][pred['scores']>MIN_SCORE])]
        pred_boxes = pred["boxes"][pred['scores']>MIN_SCORE].long()
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        masks = (pred["masks"][pred['scores']>MIN_SCORE] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


        #some criteria to select best instance...
        #example: select instance with maximum score
        best_mask = masks[torch.argmax(pred['scores']),...].cpu().numpy()

        #compute mask center in pixel coordinates
        mask_xcoord, mask_ycoord = get_mask_center(best_mask)

        #map mask center in pixel coordinates to 3D point coordinate
        mask_3Dcoord = camera.deproject_pixel([mask_xcoord,mask_ycoord],aligned_depth_frame)

        #compute surface normal at mask
        vert_mask, color_mask = mask_vertices_colors(best_mask, color_image, aligned_depth_frame)
        surface_normal = get_surface_normal(vert_mask)

        output_image = cv2.circle(output_image.permute(1, 2, 0).cpu().numpy(), (mask_xcoord, mask_ycoord), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense2', output_image.permute(1, 2, 0).cpu().numpy())
        cv2.imshow('RealSense2', output_image)
        cv2.waitKey(1)


    
