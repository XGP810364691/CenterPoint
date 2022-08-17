import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

# from det3d import __version__, torchie
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator
from tools.nusc_tracking import cp_tracker as cpTracker

from tracking import Mot3D as mot
import tracking.addbox as ab


class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None

    def initialize(self):
        self.read_config()

    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def run(self, points):
        t_t = time.time()
        # print(f"input points shape: {points.shape}")
        num_features = 5
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0  # timestamp value

        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]

        # print(f"output: {outputs}")

        torch.cuda.synchronize()
        # print("  network predict time cost:", time.time() - t)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        # print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        # print(f"  total cost time: {time.time() - t_t}")

        return scores, boxes_lidar, types



if __name__ == "__main__":
    global proc
    ## CenterPoint
    # config_path = 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py'
    # model_path = 'models/last.pth'
    config_path = 'configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py'
    model_path = 'configs/nusc/pp/centerpoint_pillar.pth'

    proc_1 = Processor_ROS(config_path, model_path)
    proc_1.initialize()

    path = '/home/xgp/data/npyData/data_h4/npy/'
    pcd_list = os.listdir(path)
    data_dict = {}
    data_dict['batch_size'] = 1
    for i in range(len(pcd_list)):
        fn = path + str(i) + '.npy'
        points = np.load(fn)
        if points.shape[1] == 4:
            o = np.zeros((len(points), 1)).astype('float32')
            points = np.column_stack((points, o))
        points = points[np.where(~np.isnan(points[:, 0]))]

        scores, dt_box_lidar, types = proc_1.run(points)
        result = np.column_stack((dt_box_lidar, types, scores))
        result[:, 8] = -result[:, 8] - np.pi / 2
        fns = '/home/xgp/data/npyData/data_h4/cp_result/' + str(i) + '.txt'
        np.savetxt(fns, result, fmt='%.3f')
        print(fns)
        jk = 0