import os
from datasets.temporal_dataset_base import TemporalDatasetBase
from scipy.spatial.transform import Rotation
import pickle
from collections import defaultdict
import numpy as np
import glob
import json
from nvtorchcam.cameras import Kitti360FisheyeCamera
from nvtorchcam.utils import normalized_intrinsics_from_pixel_intrinsics
import torch
import torch.nn as nn
import imageio
import cv2
from typing import Optional, Tuple
import re
import yaml


class Kitti360TemporalDataset(TemporalDatasetBase):

    def __init__(self,  dataset_path: str = '/home/daniel/Downloads/KITTI-360',
                        scene_name: str = '2013_05_28_drive_0007_sync',
                        image_num: str = 'image_02', 
                        init_resize: Optional[Tuple[int,int]] = None, 
                        num_forward_context: int = 1,
                        num_backward_context: int = 1,
                        mode: str = 'index_context',
                        forward_index_context_step: Optional[int] = None,
                        backward_index_context_step: Optional[int] = None,
                        nominal_forward_context_distance: Optional[float] =None,
                        forward_look_ahead: Optional[int] = None,
                        nominal_backward_context_distance: Optional[float] =None,
                        backward_look_ahead: Optional[int] = None,
                        normalize_trans: bool = True,
                        start_ref_frames_remove: int = 0,
                        end_ref_frames_remove: int = 0,
                        distance_filter_threshold: Optional[float] = None,
                        gpu_transforms: Optional[nn.Module] = None,
                    ):
        super().__init__(num_forward_context,
                        num_backward_context,
                        mode,
                        forward_index_context_step,
                        backward_index_context_step,
                        nominal_forward_context_distance,
                        forward_look_ahead,
                        nominal_backward_context_distance,
                        backward_look_ahead,
                        normalize_trans,
                        start_ref_frames_remove=start_ref_frames_remove,
                        end_ref_frames_remove=end_ref_frames_remove,
                        distance_filter_threshold=distance_filter_threshold,
                        gpu_transforms=gpu_transforms)

        self.init_resize = init_resize
        cam0_to_world = np.loadtxt(os.path.join(dataset_path,'data_poses',scene_name,'cam0_to_world.txt'))
        poses = np.loadtxt(os.path.join(dataset_path,'data_poses',scene_name,'poses.txt'))
        cam_to_pose = loadCalibrationCameraToPose(os.path.join(dataset_path,'calibration','calib_cam_to_pose.txt'))
        self.cam_to_pose = cam_to_pose[image_num]
        
        self.image_path = os.path.join(dataset_path,'data_2d_raw',scene_name,image_num,'data_rgb')

        
        N = poses.shape[0]
        frame_list = []
        #convert poses to world
        for i in range(N):
            pose = poses[i,1:].reshape(3,4)
            pose = np.concatenate([pose,np.array([[0,0,0,1]])],axis=0)
            to_world = np.matmul(pose,self.cam_to_pose)
            frame = {'to_world': to_world.astype(np.float32), 'image_path': '%010d.png' % cam0_to_world[i,0]}
            frame_list.append(frame)
        
        seq = {'frame_list': frame_list}
        self.sequence_dict = {scene_name + '_' + image_num: seq}
        self.setup()
      


    
        intrinsics = readYAMLFile(os.path.join(dataset_path,'calibration',image_num+'.yaml'))
        h_w = (intrinsics['image_height'], intrinsics['image_width'])
        

        fx = torch.tensor(intrinsics['projection_parameters']['gamma1'])
        fy = torch.tensor(intrinsics['projection_parameters']['gamma2'])
        cx = torch.tensor(intrinsics['projection_parameters']['u0'])
        cy = torch.tensor(intrinsics['projection_parameters']['v0'])
        k1 = torch.tensor(intrinsics['distortion_parameters']['k1'])
        k2 = torch.tensor(intrinsics['distortion_parameters']['k2'])
        xi = torch.tensor(intrinsics['mirror_parameters']['xi'])

        K =torch.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        K = normalized_intrinsics_from_pixel_intrinsics(K,h_w)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2] 
   
        self.camera = Kitti360FisheyeCamera.make(K, k1, k2, xi, np.pi/2)
        
    def load_frame(self, sequence_key, frame_idx, index_in_sample):
        seq = self.sequence_dict[sequence_key]
        frame = seq['frame_list'][frame_idx]
        
        sample = {}

        im_path = os.path.join(self.image_path,frame['image_path'])
        image = imageio.imread(im_path).astype(np.float32)/255
        if self.init_resize is not None:
            image = cv2.resize(image,(self.init_resize[1],self.init_resize[0]),interpolation=cv2.INTER_LINEAR)
            
        image = torch.from_numpy(image).permute(2,0,1)

    
        distance = torch.full((1,*image.shape[1:3]), fill_value=torch.nan)

        to_world = torch.from_numpy(frame['to_world'])
        camera = self.camera
        sample['image'] = image
        sample['distance'] = distance
        sample['camera'] = camera
        sample['to_world'] = to_world
        return sample
    
def readYAMLFile(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.safe_load(yamlFileOut)
    return ret


def loadCalibrationCameraToPose(filename):
    # check file
    #checkfile(filename)

    # open file
    fid = open(filename,'r')
     
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
      
    # close file
    fid.close()
    return Tr

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)
    
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None
    
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

