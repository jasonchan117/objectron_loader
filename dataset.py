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
from PIL import Image
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

flag = 1
def generate_depth(cate, img_size = (480, 640), mode = 'test', root = '/media/lang/My Passport/Dataset/Objectron/videos/'):
    global flag
    print('>>', cate, mode)
    infer_helper = InferenceHelper(dataset='nyu', device='cpu')
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
        frames = grab_frame_ffmeg(video, [i for i in range(int(num_frames))], img_size = img_size)
        depth_dir = os.path.join(root, mode, cate, i.split('.')[0])
        os.makedirs(depth_dir, exist_ok = True)
        for ind, i in enumerate(frames):
            _, predicted_depth = infer_helper.predict_pil(i)
            predicted_depth = np.squeeze(predicted_depth, 0)
            predicted_depth = np.transpose(predicted_depth, (1, 2, 0))
            cv.imwrite(os.path.join(depth_dir, str(ind)+'.png'), predicted_depth)

class Dataset(data.Dataset):
    def __init__(self, mode, cate, img_size = (480, 640), length = 5000 ,b_len = 4, root = '/media/lang/My Passport/Dataset/Objectron/videos/', history = True, priori3d = 'pointcloud'):
        self.mode = mode
        self.history = history
        self.root = root
        self.cate = cate
        self.b_len = b_len
        self.video_names = []
        self.img_size = img_size
        self.length = length
        # self.infer_helper = InferenceHelper(dataset='nyu', device='cpu')

        file_name_list = os.listdir(os.path.join(self.root, mode, self.cate))
        for i in file_name_list:
            if i[-1] == 'V' and len(i) != 4:
                self.video_names.append(i)
        self.frames_name_list = []
        self.frame_ids_list = []
        self.b_frame_ids_list = []
        self.frames = []
        self.b_frames = []
        self.num_frames = []


        for i in range(length):
            print('Sample:{}'.format(i + 1))
            choose_video = random.sample(self.video_names, 1)[0]
            name = choose_video.split('.')[0]
            annotation_name = name + '-annotation.pbdata'
            # sequence_geometry = get_geometry_data(geometry_data)
            annotation = os.path.join(self.root, self.mode, self.cate, annotation_name)
            annotation_data, instances = get_frame_annotation(annotation)
            num_frames = len(annotation_data)
            self.num_frames.append(num_frames)
            choose_frames = random.sample([i for i in range(self.b_len, num_frames)], 2)
            choose_frames = sorted(choose_frames)
            from_frame_id = choose_frames[0]
            to_frame_id = choose_frames[1]
            frame_ids = [from_frame_id, to_frame_id]
            #os.path.join(os.path.join(self.root, self.mode, self.cate,
            self.frames_name_list.append(choose_video)
            frames = grab_frame(os.path.join(os.path.join(self.root, self.mode, self.cate, choose_video)), frame_ids, img_size=self.img_size) # The list that stores the RGB of current frame and next frame
            self.frames.append(frames)
            self.frame_ids_list.append(frame_ids)
            if self.history != True:
                b_frame_ids = [i for i in range(from_frame_id-self.b_len, from_frame_id)]
                # The list that stores the RGB image of the frames in the memory bank.
                b_frames = grab_frame(os.path.join(os.path.join(self.root, self.mode, self.cate, choose_video)), b_frame_ids, img_size = self.img_size)
                self.b_frames.append(b_frames)
                self.b_frame_ids_list.append(b_frame_ids)
            # self.frames_depth_list.append([np.squeeze(self.infer_helper.predict_pil(frames[0])[1]) * 1000, np.squeeze(self.infer_helper.predict_pil(frames[1])[1])*1000])

            # temp = []
            # for j in range(self.b_len):
            #     temp.append(np.squeeze(self.infer_helper.predict_pil(b_frames[i])[1]) * 1000)
            # self.b_frames_depth_list.append(temp)


    def __len__(self):
        return self.length
    def get_2dBb(self, points_2d):
        # Finding the 2d bounding box of the image
        points_2d = np.array(points_2d)

        return
    def __getitem__(self, index):
        '''

        :param index:
        :return: RGB and Depth of the current and next frames. The RGB and Depth of the frames in memory bank. Transformation matrix of current frame and next frame. The vertices of 3D bounding boxes
                of current frame and next frame
        '''


        choose_video = self.frames_name_list[index]
        # choose_video = self.video_names[0]
        name = choose_video.split('.')[0]
        geometry_name = name + '-geometry.pbdata'
        annotation_name = name + '-annotation.pbdata'
        geometry_data = os.path.join(self.root, self.mode, self.cate, geometry_name)

        sequence_geometry = get_geometry_data(geometry_data)
        annotation = os.path.join(self.root, self.mode, self.cate, annotation_name)
        annotation_data, instances = get_frame_annotation(annotation)
        num_frames = len(annotation_data)

        # Get current frame and next frame.
        frames = self.frames[index]
        # Get the frames in the memory bank.
        if self.history != True:
            b_frames = self.b_frames[index]
        frame_depth = []
        # Get the depth of current frame and next frame [(1, 480, 640), (1, 480, 640)]
        frame_depth_dir = os.path.join(self.root, self.mode, self.cate, name)
        frame_depth.append(np.transpose(np.array(cv.imread(os.path.join(frame_depth_dir, str(self.frame_ids_list[index][0]) + '.png'))) * 1000, (2, 0, 1))[0])
        frame_depth.append(np.transpose(np.array(cv.imread(os.path.join(frame_depth_dir, str(self.frame_ids_list[index][1]) + '.png'))) * 1000, (2, 0, 1))[0])

        # Get the depth of the frames in the memory bank.
        if self.history != True:
            b_frame_depths = []
            for j in range(self.b_len):
                b_frame_depths.append(np.transpose(np.array(cv.imread(os.path.join(frame_depth_dir, str(self.frame_ids_list[index][j]) + '.png'))) * 1000, (2, 0, 1)))

        # Make the selected instance is the same in one sample.
        choose_instance = 0
        flag = 0
        # The list that store all the transformation matrix for each single frame
        transform_mats = []
        points_2d_list = []
        for i in range(len(self.frame_ids_list[index])):

            frame_id = self.frame_ids_list[index][i]
            image = frames[i]
            height, width, _ = image.shape # height: 1920, width = 1440 after resize 640, 480
            points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[frame_id]
            num_instances = len(num_keypoints)
            if num_instances > 1 and flag == 0:
                choose_instance = random.sample([i for i in range(num_instances)], 1)[0]
                flag = 1

            points_2d = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
            points_2d = [points.reshape(-1, 3) for points in points_2d]
            points_2d = [
                np.multiply(keypoint, np.asarray([width, height, 1.], np.float32)).astype(int)
                for keypoint in points_2d
            ]


            # Now, let's compute the box's vertices in 3D, then project them back to 2D:
            # for instance_id in range(num_instances):
                # The annotation contains the box's transformation and scale in world coordinate system
                # Here the instance_vertices_3d are the box vertices in the "BOX" coordinate, (i.e. it's a unit box)
                # and has to be transformed to the world coordinate.

            # Grab a randomly selected instance from one frame.
            instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[choose_instance]

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
            transform_mats.append(frame_view_matrix)
            # Project the points
            points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
            points2d_ndc = points2d_ndc.T

            # Convert the 2D Projected points from the normalized device coordinates to pixel values
            x = points2d_ndc[:, 1]
            y = points2d_ndc[:, 0]
            points2d = np.copy(points2d_ndc)
            points2d[:, 0] = ((1 + x) * 0.5) * width
            points2d[:, 1] = ((1 + y) * 0.5) * height
            points_2d_list.append(points2d.astype(int))
        # print(np.array(frames).shape, np.array(transform_mats).shape, np.array(frame_depth).shape, np.array(points_2d_list).shape)
        # Shapes: frame: (3, 640, 480), transformation matrix : (4, 4), depth: (640, 480), points_2d_list: (9, 3)
        return (np.array(np.transpose(frames[0], (2, 1, 0)))/ 255.).astype(np.float32),(np.array(np.transpose(frames[1], (2, 1, 0)))/ 255.).astype(np.float32), np.array(transform_mats[0]).astype(np.float32),np.array(transform_mats[1]).astype(np.float32) , np.array(frame_depth[0]).astype(np.float32), np.array(frame_depth[1]).astype(np.float32), np.array(points_2d_list[0]).astype(np.float32), np.array(points_2d_list[1]).astype(np.float32)



d = Dataset(mode = 'train', cate = 'laptop', length= 5)
dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True, num_workers=0)
for i, data in enumerate(dataloader, 0):

    frame1, frame2, mat1, mat2, depth1, depth2, point2d_1, point2d_2 = data
    print(frame1.shape, mat1.shape, depth1.shape, depth2.shape, point2d_1.shape)



# cates = ['laptop', 'shoe', 'cup', 'camera', 'bottle', 'book', 'chair', 'cereal_box', 'bike']
# for i in cates:
#     generate_depth(i, mode = 'train')




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