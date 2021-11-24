import numpy as np
import numpy as np
import os
import struct
import sys
import subprocess
import cv2
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
import objectron.dataset.box as Box
from IPython.core.display import display,HTML
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import _pickle as cPickle
from skimage.transform import resize
import matplotlib.pyplot as plt
from AdaBins.infer import InferenceHelper
import random
import sys
import os
from AdaBins.infer import InferenceHelper

from dataset_utils import *


def generate_depth(cate, img_size = (480, 640), mode = 'test', root = '/media/lang/My Passport/Dataset/Objectron/videos/'):
    print('>>', cate, mode)
    infer_helper = InferenceHelper(dataset='nyu')
    file_name_list = os.listdir(os.path.join(root, mode, cate))
    video_names = []
    for i in file_name_list:
        if i[-1] == 'V' and len(i) != 4:
            video_names.append(i)
    for i in video_names:
        print('->', i)
        video = os.path.join(root, mode, cate, i)
        capture = cv2.VideoCapture(video)
        num_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        frames = grab_frame(video, [i for i in range(int(num_frames))], img_size = img_size)
        depth_dir = os.path.join(root, mode, cate, i.split('.')[0])
        os.makedirs(depth_dir, exist_ok = True)
        for ind, i in enumerate(frames):
            _, predicted_depth = infer_helper.predict_pil(i)
            predicted_depth = np.squeeze(predicted_depth, 0)
            predicted_depth = np.transpose(predicted_depth, (1, 2, 0))
            cv.imwrite(os.path.join(depth_dir, str(ind)+'.png'), predicted_depth)

class Dataset(data.Dataset):
    def __init__(self, mode, cate, img_size = (480, 640), length = 5000 ,b_len = 4, root = '/media/lang/My Passport/Dataset/Objectron/videos/'):
        self.mode = mode
        self.root = root
        self.cate = cate
        self.b_len = b_len
        self.video_names = []
        self.img_size = img_size
        self.length = length
        file_name_list = os.listdir(os.path.join(self.root, mode, self.cate))
        for i in file_name_list:
            if i[-1] == 'V' and len(i) != 4:
                self.video_names.append(i)

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        choose_video = random.sample(self.video_names, 1)[0]
        # choose_video = self.video_names[0]
        name = choose_video.split('.')[0]
        geometry_name = name + '-geometry.pbdata'
        annotation_name = name + '-annotation.pbdata'
        geometry_data = os.path.join(os.path.join(self.root, self.mode, self.cate, geometry_name))

        sequence_geometry = get_geometry_data(geometry_data)
        annotation = os.path.join(os.path.join(self.root, self.mode, self.cate, annotation_name))
        annotation_data, instances = get_frame_annotation(annotation)
        num_frames = len(annotation_data)

        while True:
            choose_frames = random.sample([i for i in range(self.b_len, num_frames)], 2)
            choose_frames = sorted(choose_frames)
            from_frame_id = choose_frames[0]
            to_frame_id = choose_frames[1]
            frame_ids = [from_frame_id, to_frame_id]

            frames = grab_frame(os.path.join(os.path.join(self.root, self.mode, self.cate, choose_video)), frame_ids, img_size=self.img_size)

            b_frame_ids = [i for i in range(from_frame_id-self.b_len, from_frame_id)]
            b_frames = grab_frame(os.path.join(os.path.join(self.root, self.mode, self.cate, choose_video)), b_frame_ids, img_size = self.img_size)
            if b_frames != -1 and frames != -1:
                break


        for i in range(len(frame_ids)):
            frame_id = frame_ids[i]
            image = frames[i]
            height, width, _ = image.shape # height: 1920, width = 1440 after resize 640, 480
            print(height, width)
            points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[frame_id]
            num_instances = len(num_keypoints)
            points_2d = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
            points_2d = [points.reshape(-1, 3) for points in points_2d]
            points_2d = [
                np.multiply(keypoint, np.asarray([width, height, 1.], np.float32)).astype(int)
                for keypoint in points_2d
            ]

            points_2d = []
            # Now, let's compute the box's vertices in 3D, then project them back to 2D:
            for instance_id in range(num_instances):
                # The annotation contains the box's transformation and scale in world coordinate system
                # Here the instance_vertices_3d are the box vertices in the "BOX" coordinate, (i.e. it's a unit box)
                # and has to be transformed to the world coordinate.
                instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[instance_id]

                box_transformation = np.eye(4)
                box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
                box_transformation[:3, -1] = instance_translation
                vertices_3d = instance_vertices_3d * instance_scale.T
                # Homogenize the points
                vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
                # Transform the homogenious 3D vertices with the box transformation
                box_vertices_3d_world = np.matmul(box_transformation, vertices_3d_homg)

                # If we transform these vertices to the camera frame, we get the 3D keypoints in the annotation data
                # i.e. vertices_3d_cam == points_3d
                vertices_3d_cam = np.matmul(frame_view_matrix, box_vertices_3d_world)
                vertices_2d_proj = np.matmul(frame_projection_matrix, vertices_3d_cam)

                # Project the points
                points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
                points2d_ndc = points2d_ndc.T

                # Convert the 2D Projected points from the normalized device coordinates to pixel values
                x = points2d_ndc[:, 1]
                y = points2d_ndc[:, 0]
                points2d = np.copy(points2d_ndc)
                points2d[:, 0] = ((1 + x) * 0.5) * width
                points2d[:, 1] = ((1 + y) * 0.5) * height
                points_2d.append(points2d.astype(int))

        return 0

# d = Dataset('train', 'laptop')
# dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True, num_workers=0)
# for i, data in enumerate(dataloader, 0):
#     print(data)

cates = ['laptop', 'shoe', 'cup', 'camera', 'bottle', 'book', 'chair', 'cereal_box', 'bike']
for i in cates:
    generate_depth(i)




sys.exit()

index = '3-13'
video = '/media/lang/My Passport/Dataset/Objectron/videos/test/bike/bike-batch-'+index + '.MOV'
frames = grab_frame(video,[0, 1, 2, 3])
print(frames[0].shape)
laptop = '/media/lang/My Passport/Dataset/Objectron/videos/test/bike/bike-batch-' + index + '-geometry.pbdata'
annotation = '/media/lang/My Passport/Dataset/Objectron/videos/test/bike/bike-batch-'+ index + '-annotation.pbdata'

frame_num = 100

print('Start')


####################################################################

cap = cv2.VideoCapture(video)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

frame_ids = [0, 10, 50, 100]
num_frames = len(frame_ids)
sequence_geometry = get_geometry_data(laptop)
frames = grab_frame(video, frame_ids)
annotation_data, instances = get_frame_annotation(annotation)
fig, ax = plt.subplots(1, num_frames, figsize = (12, 16))

## Authors design
for i in range(len(frame_ids)):
    frame_id = frame_ids[i]
    image = frames[i]
    height, width, _ = image.shape

    points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[frame_id]
    num_instances = len(num_keypoints)

    # As covered in our previous tutorial, we can directly grab the 2D projected points from the annotation
    # file. The projections are normalized, so we scale them with the image's height and width to get
    # the pixel value.
    # The keypoints are [x, y, d] where `x` and `y` are normalized (`uv`-system)\
    # and `d` is the metric distance from the center of the camera. Convert them
    # keypoint's `xy` value to pixel.
    points_2d = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
    points_2d = [points.reshape(-1, 3) for points in points_2d]
    points_2d = [
        np.multiply(keypoint, np.asarray([width, height, 1.], np.float32)).astype(int)
        for keypoint in points_2d
    ]

    points_2d = []
    # Now, let's compute the box's vertices in 3D, then project them back to 2D:

    for instance_id in range(num_instances):
        # The annotation contains the box's transformation and scale in world coordinate system
        # Here the instance_vertices_3d are the box vertices in the "BOX" coordinate, (i.e. it's a unit box)
        # and has to be transformed to the world coordinate.
        instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[instance_id]

        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation
        vertices_3d = instance_vertices_3d * instance_scale.T;
        # Homogenize the points
        vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
        # Transform the homogenious 3D vertices with the box transformation
        box_vertices_3d_world = np.matmul(box_transformation, vertices_3d_homg)

        # If we transform these vertices to the camera frame, we get the 3D keypoints in the annotation data
        # i.e. vertices_3d_cam == points_3d
        vertices_3d_cam = np.matmul(frame_view_matrix, box_vertices_3d_world)
        vertices_2d_proj = np.matmul(frame_projection_matrix, vertices_3d_cam)

        # Project the points
        points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
        points2d_ndc = points2d_ndc.T

        # Convert the 2D Projected points from the normalized device coordinates to pixel values
        x = points2d_ndc[:, 1]
        y = points2d_ndc[:, 0]
        points2d = np.copy(points2d_ndc)
        points2d[:, 0] = ((1 + x) * 0.5) * width
        points2d[:, 1] = ((1 + y) * 0.5) * height
        points_2d.append(points2d.astype(int))
        # points2d are the projected 3D points on the image plane.

    # Visualize the boxes
    for instance_id in range(num_instances):
        for kp_id in range(num_keypoints[instance_id]):
            kp_pixel = points_2d[instance_id][kp_id, :]
            cv2.circle(image, (kp_pixel[0], kp_pixel[1]), 10, (255, 0, 0), -1)
        for edge in Box.EDGES:
            start_kp = points_2d[instance_id][edge[0], :]
            end_kp = points_2d[instance_id][edge[1], :]
            cv2.line(image, (start_kp[0], start_kp[1]), (end_kp[0], end_kp[1]), (255, 0, 0), 2)

    # We can also use the above pipeline to visualize the scene point-cloud on the image.
    # First, let's grab the point-cloud from the geometry metadata
    transform, projection, view, scene_points_3d = sequence_geometry[frame_id]
    scene_points_2d = project_points(scene_points_3d, projection, view, width, height)
    # Note these points all have estimated depth, which can double as a sparse depth map for the image.

    for point_id in range(scene_points_2d.shape[0]):
        cv2.circle(image, (scene_points_2d[point_id, 0], scene_points_2d[point_id, 1]), 10,
                   (0, 255, 0), -1)
    ax[i].grid(False)
    ax[i].imshow(image);
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)

fig.tight_layout();
plt.show()

sys.exit()
points_t = []
transforms_matrix = []
for i in range(len(frame_ids)):

    frame_id = frame_ids[i]
    print(frame_id)
    image = frames[i]
    height, width, _ = image.shape
    print(height, width)
    # The frame_view_matrix is the inverse of transformation matrix.
    points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[frame_id]

    num_instances = len(num_keypoints)

    # As covered in our previous tutorial, we can directly grab the 2D projected points from the annotation
    # file. The projections are normalized, so we scale them with the image's height and width to get
    # the pixel value.
    # The keypoints are [x, y, d] where `x` and `y` are normalized (`uv`-system)\
    # and `d` is the metric distance from the center of the camera. Convert them
    # keypoint's `xy` value to pixel.
    points_2d = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
    points_2d = [points.reshape(-1, 3) for points in points_2d]
    points_2d = [
        np.multiply(keypoint, np.asarray([width, height, 1.], np.float32)).astype(int)
        for keypoint in points_2d
    ]

    points_2d = []
    # Now, let's compute the box's vertices in 3D, then project them back to 2D:
    print('num:',num_instances)
    for instance_id in range(num_instances):
        # The annotation contains the box's transformation and scale in world coordinate system
        # Here the instance_vertices_3d are the box vertices in the "BOX" coordinate, (i.e. it's a unit box)
        # and has to be transformed to the world coordinate.
        instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[instance_id] # instance_vertices_3d is the coordinates of vertices of the bounding box in box coordinate.

        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation

        vertices_3d = instance_vertices_3d * instance_scale.T;

        # Homogenize the points
        vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T

        # Transform the homogenious 3D vertices with the box transformation
        box_vertices_3d_world = np.matmul(box_transformation, vertices_3d_homg)# Transform the vertices back to world coordinate.

        # If we transform these vertices to the camera frame, we get the 3D keypoints in the annotation data
        # i.e. vertices_3d_cam == points_3d
        vertices_3d_cam = np.matmul(frame_view_matrix, box_vertices_3d_world)# transform the vertices in world coordinate to camera coordinate.
        # print(np.transpose(np.array(vertices_3d_cam),(1, 0)), np.array(vertices_3d_cam).shape)
        transforms_matrix.append(frame_view_matrix)
        points_t.append(np.transpose(np.array(vertices_3d_cam),(1, 0))[6])
        vertices_2d_proj = np.matmul(frame_projection_matrix, vertices_3d_cam)

        # Project the points
        points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
        points2d_ndc = points2d_ndc.T
        # Convert the 2D Projected points from the normalized device coordinates to pixel values
        x = points2d_ndc[:, 1]
        y = points2d_ndc[:, 0]
        points2d = np.copy(points2d_ndc)
        points2d[:, 0] = ((1 + x) * 0.5) * width
        points2d[:, 1] = ((1 + y) * 0.5) * height
        points_2d.append(points2d.astype(int))
    # Visualize the boxes
    for instance_id in range(num_instances):
        for kp_id in range(num_keypoints[instance_id]):
            kp_pixel = points_2d[instance_id][kp_id, :]
            cv2.circle(image, (kp_pixel[0], kp_pixel[1]), 10, (255, 0, 0), -1)
        for edge in Box.EDGES:
            start_kp = points_2d[instance_id][edge[0], :]
            end_kp = points_2d[instance_id][edge[1], :]
            cv2.line(image, (start_kp[0], start_kp[1]), (end_kp[0], end_kp[1]), (255, 0, 0), 2)

    # We can also use the above pipeline to visualize the scene point-cloud on the image.
    # First, let's grab the point-cloud from the geometry metadata
    transform, projection, view, scene_points_3d = sequence_geometry[frame_id]
    scene_points_2d = project_points(scene_points_3d, projection, view, width, height)
    # Note these points all have estimated depth, which can double as a sparse depth map for the image.

    for point_id in range(scene_points_2d.shape[0]):
        cv2.circle(image, (scene_points_2d[point_id, 0], scene_points_2d[point_id, 1]), 10,
                   (0, 255, 0), -1)
    ax[i].grid(False)
    ax[i].imshow(image);
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)

fig.tight_layout()
plt.show()


        # points2d are the projected 3D points on the image plane.
points_t = np.array(points_t)
# print(intrinsic)
change = np.dot(transforms_matrix[1], np.linalg.inv(transforms_matrix[0]))
print(points_t)
print(change)
#points_t[0] = np.dot(np.linalg.inv(intrinsic), points_t[0])
# points_t[1] = np.dot(np.linalg.inv(intrinsic), points_t[1])

# print(points_t)
# r = change[:3,:3]
# t = change[:3,3]
# res = np.dot(r, points_t[0]) + t
# print(res/res[2])
print(np.dot(change, points_t[0]))