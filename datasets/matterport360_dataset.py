import torch.utils.data as data
import torch
import numpy as np
import os
import glob
import cv2
import imageio
from struct import unpack
from scipy.spatial.transform import Rotation
from nvtorchcam.cameras import EquirectangularCamera
from typing import Union, Optional, Tuple

class Matterport360Dataset(data.Dataset):
    def __init__(self, dataset_path: str, 
                       scene_list_path: str, 
                       init_resize: Optional[Tuple[int, int]] = None, 
                       gpu_transforms: Optional[torch.nn.Module] = None, 
                       image_interp_mode: str = 'bilinear', 
                       distance_threshold: int = 2):
        
        with open(scene_list_path,'r') as f:
            self.scene_list = f.read().split()

        if image_interp_mode == 'bilinear':
            self.image_interp_mode = cv2.INTER_LINEAR
        elif image_interp_mode == 'nearest':
            self.image_interp_mode = cv2.INTER_NEAREST
        else:
            raise RuntimeError('interpolation mode not recognized')
       
        self.dataset_path = dataset_path
        scene_folders = glob.glob(os.path.join(dataset_path,'*'))
        self.scene_folders = sorted([x for x in scene_folders if os.path.isdir(x)])
        
        self.gpu_transforms = gpu_transforms
        self.init_resize = init_resize
        
        print('getting distance tables')
        self.distance_tables = self.get_distance_tables()
        print('done getting distance tables')
        self.pairs = self.get_pairs(distance_threshold)
    
        coord_change = np.eye(4)
        coord_change[1,1] = -1
        coord_change[2,2] = -1
        self.coord_change = coord_change


    def get_distance_tables(self):
        save_name = 'datasets/matterport_distance_tables.npy'
        
        if os.path.exists(save_name):
            print('skipping getting distance tables')
            return np.load(save_name,allow_pickle=True)[()]
            

        distance_tables = {}
        for scene_folder in self.scene_folders:
            scene_name = os.path.basename(scene_folder)
    
            scene_dist_table = {}
            distance_tables[scene_name] = scene_dist_table
            
            pose_paths = sorted(glob.glob(os.path.join(scene_folder,'*pose.txt')))
            scene_dist_table['names'] = [os.path.basename(x)[:-9] for x in pose_paths]
            
            poses = []
            for pp in pose_paths:
                poses.append(np.loadtxt(pp))
    
            poses = np.stack(poses)
            positions = poses[:,:3]
            #(n,3)
            positions = torch.from_numpy(positions)
            #(n,n)
            dists = torch.norm( positions.unsqueeze(0)-positions.unsqueeze(1),dim=2)
            #i = torch.arange(dists.size(0))
            #dists[i,i] = 1e8

            scene_dist_table['distance'] = dists
        np.save(save_name, distance_tables)
        return distance_tables


    def get_pairs(self, dist_threshold):
        pairs = []
        for scene in self.scene_list: # this will account for train test
            names = self.distance_tables[scene]['names']
            dists = self.distance_tables[scene]['distance']
            scene_path = os.path.join(self.dataset_path,scene)
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    if dists[i][j] < dist_threshold:
                        pairs.append( (os.path.join(scene_path,names[i]),os.path.join(scene_path,names[j])) )

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        images = []
        depths = []
        to_worlds = []

        for path in pair:
            im = imageio.imread(path+'_rgb.png').astype(np.float32)/255
            depth =  read_dpt(path+'_depth.dpt')
            im[:142,:,:] = 0
            im[-142:,:,:] = 0
            depth[:142,:] = 0
            depth[-142:,:] = 0
            if self.init_resize is not None:
                im = cv2.resize(im,(self.init_resize[1],self.init_resize[0]), interpolation = self.image_interp_mode)
                depth = cv2.resize(depth,(self.init_resize[1],self.init_resize[0]),interpolation = cv2.INTER_NEAREST)

            pose = np.loadtxt(path+'_pose.txt')
            rot = Rotation.from_quat(pose[3:])
            R = rot.as_matrix()
            t = pose[:3]
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t

            T =np.matmul(T,self.coord_change)

            images.append(im)
            depths.append(depth)
            to_worlds.append(T.astype(np.float32))

       
        images = torch.from_numpy(np.stack(images,axis=0)).permute(0,3,1,2)
        depths = torch.from_numpy(np.stack(depths,axis=0)).unsqueeze(1)
        to_worlds = torch.from_numpy(np.stack(to_worlds,axis=0))

        camera = EquirectangularCamera.make(batch_shape = (images.size(0),))
      
      
        depths[depths < 1e-6] = torch.nan
   
        sample = {}
        sample['idx'] = idx
        sample['image'] = images
        sample['distance'] = depths
        sample['camera'] = camera
        sample['to_world'] = to_worlds
      
        return sample


#from https://github.com/manurare/360monodepth/blob/main/code/python/src/utility/depthmap_utils.py
def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.
    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data



if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import random
    device='cuda'
    batch_size = 1
    save_batch_idx = 0

    dataset = Matterport360Dataset('/media/daniel/drive3/Datasets/matterport360/data','datasets/matterport_split/scenes_test.txt',init_resize=(2*128,2*256))
    for i in range(0,len(dataset),5):
        print(i)
        print(dataset.pairs[i])
        sample = dataset[i]
        image0 = sample['image'][0].permute(1,2,0)
        image1 = sample['image'][1].permute(1,2,0)
        plt.imshow(torch.cat([image0,image1],dim=1))
        plt.show()
    exit()
