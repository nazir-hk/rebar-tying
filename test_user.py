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


MIN_SCORE = 0.6

global click_x, click_y
click_x = None
click_y = None


def mouse_callback(event,x,y,flags,param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        click_x = x
        click_y = y

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

def get_surface_normal(mask_verts):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mask_verts[~np.any(mask_verts == 0, axis=1)]) #create pointcloud after discarding points at origin
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    for i in range(np.asarray(pcd.normals).shape[0]):
        if pcd.normals[i][2] > 0:
            pcd.normals[i][0] = -pcd.normals[i][0]
            pcd.normals[i][1] = -pcd.normals[i][1]
            pcd.normals[i][2] = -pcd.normals[i][2]
    
    normals = np.asarray(pcd.normals)

    o3d.visualization.draw_geometries([pcd], point_show_normal=True)    # Visualize point cloud 

    return np.sum(normals, axis=0) / normals.shape[0]


# Function to check if two masks overlap
def masks_overlap(mask1, mask2):
    return torch.any(torch.logical_and(mask1, mask2))

if __name__ == '__main__':

    camera = Camera()
    time.sleep(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load('C:/Users/anazi/Documents/digging/cross.pt', map_location=device))
    model.eval().to(device)

    eval_transform = get_transform(train=False)

    cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('RealSense2', mouse_callback)

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
        pred_masks = (pred["masks"][pred['scores']>MIN_SCORE] > 0.7).squeeze(1)

        # only keep non-overlapping masks
        unique_masks = []
        unique_masks_ids = []

        for i, mask in enumerate(pred_masks):
            is_unique = True
            for unique_mask in unique_masks:
                if masks_overlap(mask, unique_mask):
                    is_unique = False
                    break
            if is_unique:
                unique_masks.append(mask)
                unique_masks_ids.append(i)

        unique_masks = torch.stack(unique_masks)

        output_image = draw_bounding_boxes(image, torch.stack([pred_boxes[i] for i in unique_masks_ids]), [pred_labels[j] for j in unique_masks_ids], colors="red")
        output_image = draw_segmentation_masks(output_image, unique_masks, alpha=0.5, colors="blue")

        cv2.imshow('RealSense2', output_image.permute(1, 2, 0).cpu().numpy())
        cv2.waitKey(1)

        #out of the predicted masks, we will pick one specified by the user
        user_mask = None
        user_mask_id = None
        if None not in (click_x, click_y):

            for i, mask in enumerate(unique_masks):
                if mask[click_y, click_x]:
                    user_mask = mask
                    user_mask_id = unique_masks_ids[i]
                    break
                
            if None not in (user_mask, user_mask_id):
                # display user-selected mask and break the while loop
                output_image = draw_segmentation_masks(image, user_mask, alpha=0.8, colors="green")
                cv2.imshow('RealSense2', output_image.permute(1, 2, 0).cpu().numpy())
                cv2.waitKey(1)
                time.sleep(5)
                break
            else:
                # reset click parameters and continue
                click_x = None
                click_y = None

    cv2.destroyAllWindows()

    
    print("click_xcoord: ", click_x)
    print("click_ycoord: ", click_y)

    #compute mask center in pixel coordinates
    mask_xcoord, mask_ycoord = get_mask_center(user_mask.cpu().numpy())

    #map mask center in pixel coordinates to 3D point coordinate
    mask_3Dcoord = camera.deproject_pixel([mask_xcoord,mask_ycoord],aligned_depth_frame)
    print(mask_3Dcoord)

    #compute surface normal at mask
    mask_vertices, mask_vertices_color = mask_vertices_colors(user_mask.cpu().numpy(), color_image, aligned_depth_frame)
    surface_normal = get_surface_normal(mask_vertices)




    
