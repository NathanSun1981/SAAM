# SAAM
using transformer and segmentation anything for realtime segmentation

# Requirements
- Python 3.8
- CUDA 11.1+

# Installation
```bash
conda env create -f environment.yaml
conda activate saam
python -m spacy download en_core_web_sm

cd ..
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ../Semantic-Segment-Anything
```
Dowload pth of SAM

```bash
mkdir ckp && cd ckp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```
# Configuration
all default configurations are stored in default_config.yml and config.py
avaliable parameters
```bash
(block_count=40000, ckpt_path='ckp/sam_vit_h_4b8939.pth', ckpt_yolo='yolov8n-seg.pt', classes=[23, 30, 31], clipping_distance_in_meters=3.0, 
color_dir='data/LoungeRGBDImages/color', color_folder='color_output', convert_to_rle=False, crop_n_layers=1, 
crop_n_points_downscale_factor=2, dataset='ade20k', depth_dir='data/LoungeRGBDImages/depth', depth_folder='depth_output', 
depth_max=3.0, depth_min=0.1, depth_scale=1000.0, device='CUDA:0', engine='tensor', est_point_count=10000000, file_suffix='.png', 
fragment_size=100, global_registration_method='ransac', icp_distance_thr=0.07, icp_method='colored', icp_voxelsize=0.05, 
integration_mode='color', min_mask_region_area=100, multiprocessing=False, name='Default reconstruction system config', 
num_gpus=3, odometry_distance_thr=0.07, odometry_loop_interval=10, odometry_loop_weight=0.1, odometry_method='hybrid', 
output_name='output.npz', path_color_intrinsic='', path_dataset='data/LoungeRGBDImages/', path_intrinsic='', 
path_trajectory='data/LoungeRGBDImages/lounge_trajectory.log', points_per_side=64, pred_iou_thresh=0.86, registration_loop_weight=0.1, 
sam_device='cuda', sam_model_type='vit_h', save_img=True, seg_model='oneformer', stability_score_thresh=0.92, surface_weight_thr=3.0,
trunc_voxel_multiplier=8.0, using_sam=True, voxel_size=0.0058)
```



# RUN SAAM
```bash
python scripts/sam_transfomer.py --data_dir data/room/  --seg_model oneformer --dataset ade20k --convert_to_rle --using_sam --classes 23 30 31
```
you can choose which class  needs to be filtered out after segmenetation by setting '--classes'

using pytorch multiprocessing using multiple GPUs 
```bash
python scripts/sam_transfomer_mp.py --data_dir data/room/  --seg_model oneformer --dataset ade20k --convert_to_rle --using_sam --classes 23 30 31 --num_gpus 3
```