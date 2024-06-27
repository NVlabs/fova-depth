

# FoVA-Depth

This is the official code release for the paper  
Daniel Lichy, Hang Su, Abhishek Badki, Jan Kautz, and Orazio Gallo, **FoVA-Depth: Field-of-View Agnostic Depth Estimation for Cross-Dataset Generalization**, 3DV 2024.

Please check out the project page: [https://research.nvidia.com/labs/lpr/fova-depth/](https://research.nvidia.com/labs/lpr/fova-depth/)

:point_right: :point_right: Also take a look at [nvTorchCam](https://github.com/NVlabs/nvTorchCam), which implements plane-sweep volumes (PSV) and related concepts, such as sphere-sweep volumes or epipolar attention, in a way that is agnostic to the camera projection model (e.g., pinhole or fisheye). 

## Table of Contents
1. [Installation](#installation)
2. [Downloading Pretrained Checkpoints](#download-pretrained-checkpoints)
3. [Downloading Datasets](#downloading-datasets)
4. [Running](#running-the-code)
   - [Evaluation](#evaluation)
   - [Training](#training)
5. [Testing New Datasets](#testing-new-datasets)
6. [Citation](#citation)


## Installation

This project depends on Pytorch, Pytorch-Lightning, and our library [nvTorchCam](https://github.com/NVlabs/nvTorchCam).

To clone the nvTorchCam submodule, use the ```--recurse-submodules``` option when cloning this repo.

To install in a virtual environment run:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [nvdiffrast](https://nvlabs.github.io/nvdiffrast), though this is only strictly needed to interpolate when using cube maps.

## Download Pretrained Checkpoints

Download the pretrained checkpoints from [here]() and place them in the checkpoints folder. They should be:

checkpoints  
├── cube_ddad_2image.ckpt  
├── cube_ddad_3image.ckpt  
├── cube_scannet.ckpt  
├── erp_ddad_2image.ckpt  
├── erp_ddad_3image.ckpt  
└── erp_scannet.ckpt


## Downloading Datasets

Our models are trained on two pinhole datasets, Scannet (indoor) and DDAD (driving), and tested on the Equirectangular (ERP) dataset Matterport360 (indoor) and the fisheye dataset KITTI360 (driving). Below, we provide instructions for downloading these datasets.

### Scannet

Due to the unavailability of the original Scannet dataset version used in our work (prepared by the authors of [Normal-Assisted-Stereo](https://github.com/udaykusupati/Normal-Assisted-Stereo/tree/master?tab=readme-ov-file)), we recommend following the alternative setup provided in [this repository](https://github.com/tberriel/RayPatchQuerying/tree/main). This setup closely mimics the structure required by Normal-Assisted-Stereo.

Additionally, you will need to download new_orders and train-test splits from Normal-Assisted-Stereo, which we provide [here](). 

Once prepared the folder structure should look as follows:
```
scannet  
│  
├── train  
├── val  
└── new_orders  
    ├── train  
    └── val  
```
### DDAD

Dowload the DDAD dataset (train+val 257GB) from here https://github.com/TRI-ML/DDAD. Install the TRI Dataset Governance Policy (DGP) codebase as explained on the same page.

Then export the depth maps and resize the images by running the following script from the root of this repository:

```bash
python data_processing/resize_ddad.py --ddad_path path_to_ddad --resized_ddad_path output_path_to_store_resized_data
```
This make take several hours.

Once prepared the folder structure should look as follows:
```
ddad_resize  
├── 000000  
    ├── calibration  
    ├── depth  
    ├── rgb  
    └── scene_*.json  
├── 000001  
├── ...
└── 000199
```

### Matterport 360

Matterport360 can be download from here: https://researchdata.bath.ac.uk/1126/ as seven .zip files.

Once prepared the folder structure should look as follows:
```
data  
├──  1LXtFkjw3qL  
├──  1pXnuDYAj8r  
└── ...
```
### KITTI360

Kitti360 can be downloaded here: https://www.cvlibs.net/datasets/kitti-360/ 
You will need the fisheye images, Calibrations, and Vehicle poses. After extracting it should look as follows:
```
KITTI-360  
├── calibration  
    ├──  <drive_name>  
        ├── image_02  
        └── image_03  
├── data_2d_raw  
    └── calib_cam_to_pose.txt  
├── data_poses  
    ├──  <drive_name>  
        ├── cam0_to_world.txt  
        └── poses.txt  
```
Where `<drive_name>` will be something like `2013_05_28_drive_0007_sync` for example. 

## Running the Code

This project is based on Pytorch-Lighting and is thus highly configurable from the command-line. For all the following commands you can append `--print_config` to print all configurable options. These options can be overridden from the command-line or with a `.yaml` configuration file. See Pytorch-Lightings [docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details.

### Evaluation

Here we list the commands for testing our pretrained models on Matterport360 and KITTI360.

* ERP model on Matterport360

```bash
python train.py test --data configs/data_configs/matterport360.yaml --model configs/fova_depth_erp.yaml --model.init_args.network.init_args.warp_to_original_cam True --trainer.default_root_dir test_logs/matterport360_erp --model.init_args.load_state_dict checkpoints/erp_scannet.ckpt --data.init_args.test_datasets.init_args.dataset_path <path_to_matterport360_dataset>
```

* Cube model on Matterport360

```bash
python train.py test --data configs/data_configs/matterport360.yaml --model configs/fova_depth_cube.yaml --model.init_args.network.init_args.warp_to_original_cam True --trainer.default_root_dir test_logs/matterport360_cube --model.init_args.load_state_dict checkpoints/cube_scannet.ckpt --data.init_args.test_datasets.init_args.dataset_path <path_to_matterport360_dataset>
```

* 2-image ERP model on KITTI360

```bash
python train.py test --data configs/data_configs/kitti360.yaml --model configs/fova_depth_erp_highres.yaml --model.init_args.load_state_dict checkpoints/erp_ddad_2image.ckpt --trainer.default_root_dir test_logs/kitti360_erp --data.init_args.test_datasets.init_args.dataset_path <path_to_kitti360_dataset> --data.init_args.test_datasets.init_args.scene_name <kitti360_scene_name>
```

This saves the data in the canonical representation. It is possible to warp the depth back to the original fisheye representation by adding the following arguments:  `--model.init_args.network.init_args.warp_to_original_cam True` and `--trainer.inference_mode False`. However these will slow down inference due to iterative undistortion.

* 2-image Cubemap model on KITTI360

```bash
python train.py test --data configs/data_configs/kitti360.yaml --model configs/fova_depth_cube_highres.yaml --model.init_args.load_state_dict checkpoints/cube_ddad_2image.ckpt --trainer.default_root_dir test_logs/kitti360_cube --data.init_args.test_datasets.init_args.dataset_path <path_to_kitti360_dataset> --data.init_args.test_datasets.init_args.scene_name <kitti360_scene_name>
```

* 3-image ERP model on KITTI360

```bash
python train.py test --data configs/data_configs/kitti360_3image.yaml --model configs/fova_depth_erp_highres.yaml --model.init_args.load_state_dict checkpoints/erp_ddad_3image.ckpt --trainer.default_root_dir test_logs/kitti360_erp_3image --data.init_args.test_datasets.init_args.dataset_path <path_to_kitti360_dataset> --data.init_args.test_datasets.init_args.scene_name <kitti360_scene_name>
```

* 3-image Cube model on KITTI360

```bash
python train.py test --data configs/data_configs/kitti360_3image.yaml --model configs/fova_depth_cube_highres.yaml --model.init_args.load_state_dict checkpoints/cube_ddad_3image.ckpt --trainer.default_root_dir test_logs/kitti360_cube_3image --data.init_args.test_datasets.init_args.dataset_path <path_to_kitti360_dataset> --data.init_args.test_datasets.init_args.scene_name <kitti360_scene_name>
```

### Training

All models were trained on 8 NVIDIA V100 GPUs with 32GB of memory. Batch-sizes and learning rates may need to be adjusted when training on different hardware. Here are the commands to train the models.

* ERP model on ScanNet

```bash
python train.py fit --data configs/data_configs/scannet.yaml --model configs/fova_depth_erp.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/erp_scannet --data.init_args.train_dataset.init_args.dataset_path <path_to_scannet_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_scannet_dataset>
```

* Cube model on ScanNet

```bash
python train.py fit --data configs/data_configs/scannet.yaml --model configs/fova_depth_cube.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/cube_scannet --data.init_args.train_dataset.init_args.dataset_path <path_to_scannet_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_scannet_dataset>
```

* ERP model on DDAD (2 input images)

```bash
python train.py fit --data configs/data_configs/ddad.yaml --model configs/fova_depth_erp_highres.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/erp_ddad --model.init_args.load_state_dict checkpoints/erp_scannet.ckpt --trainer.max_epochs 40 --data.init_args.train_dataset.init_args.dataset_path <path_to_ddad_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_ddad_dataset>
```

* Cube model on DDAD (2 input images)

```bash
python train.py fit --data configs/data_configs/ddad.yaml --model configs/fova_depth_cube_highres.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/cube_ddad --model.init_args.load_state_dict checkpoints/cube_scannet.ckpt -trainer.max_epochs 40 --data.init_args.train_dataset.init_args.dataset_path <path_to_ddad_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_ddad_dataset>
```

* ERP model on DDAD (3 input images)

```bash
python train.py fit --data configs/data_configs/ddad_3image.yaml --model configs/fova_depth_erp_highres.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/erp_ddad_3image --model.init_args.load_state_dict checkpoints/erp_ddad_2image.ckpt --trainer.max_epochs 40 --model.init_args.optimizer_config.init_lr 0.00002 --data.init_args.train_dataset.init_args.dataset_path <path_to_ddad_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_ddad_dataset>
```

* Cube model on DDAD (3 input images)

```bash
python train.py fit --data configs/data_configs/ddad_3image.yaml --model configs/fova_depth_cube_highres.yaml --trainer configs/default_trainer.yaml --trainer.default_root_dir train_logs/cube_ddad_3image --model.init_args.load_state_dict checkpoints/cube_ddad_2image.ckpt --trainer.max_epochs 40 --model.init_args.optimizer_config.init_lr 0.00002 -data.init_args.train_dataset.init_args.dataset_path <path_to_ddad_dataset> --data.init_args.val_datasets.init_args.dataset_path <path_to_ddad_dataset>
```

## Testing (New) Datasets

We include some facilities for testing new datasets one might want to implement. For example, running

```bash
python datasets/test_dataset.py --data configs/data_configs/matterport360.yaml --type_to_test test --sample_number 25 --canon_type erp --data.init_args.test_datasets.init_args.dataset_path  <path_to_matterport_dataset>
```

will save the 25th sample from the Matterport training dataset to the `test_dataset_output` folder. The sample contains the original images and unprojected distance maps in world coordinates, saved in PLY format for visualization in MeshLab or similar tools to ensure alignment (i.e. you loaded all coordinate systems correctly). It also exports images warped to `--canon_type=erp` and the corresponding unprojected canonical distances in PLY. Additionally, the script saves the reference image rectified alongside each source image in ERP format, where corresponding features are vertically aligned, aiding in pose verification without needing ground truth distance.

## Citation

```bibtex
@inproceedings{lichy2024fova,
  title     = {{FoVA-Depth}: {F}ield-of-View Agnostic Depth Estimation for Cross-Dataset Generalization},
  author    = {Lichy, Daniel and Su, Hang and Badki, Abhishek and Kautz, Jan and Gallo, Orazio},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2024}
}
```