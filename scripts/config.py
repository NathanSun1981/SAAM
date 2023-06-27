# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import os

import configargparse


class ConfigParser(configargparse.ArgParser):

    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__), 'default_config.yml')
        ],
                         conflict_handler='resolve')

        # yapf:disable
        # Default arguments
        
        self.add_argument('--color_dir', default='data/LoungeRGBDImages/color', help='specify the root path of images and masks')
        self.add_argument('--depth_dir', default='data/LoungeRGBDImages/depth', help='specify the root path of images and masks')
        self.add_argument('--sam_device', default='cpu', help='The device to run generation on.')
        self.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
        self.add_argument('--ckpt-yolo', default='yolov8n-seg.pt', help='specify yolo segmentation model')
        self.add_argument('--save_img', default=True, action='store_true', help='whether to save annotated images')
        self.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'coco'], help='specify the set of class names')
        self.add_argument('--seg_model', type=str, default='oneformer', choices=['oneformer', 'segformer','yolo'], help='specify the segmenta model')
        
        self.add_argument("--file_suffix", type=str, default='.png', help='suffix of images in dataset')
        self.add_argument('--using_sam',  action="store_true", help='use sam for masks generation.')
        self.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        self.add_argument('--num_gpus', type=int, default=0, help='number of gpus')
        
        self.add('--output_name',
               help='path to the npz file that stores voxel block grid.',
               default='output.npz')
        ###argment for SAM, inherit from SAM code###
        sam_parse = self.add_argument_group('sam')
        sam_parse.add_argument("--sam_model_type", type=str, default='vit_h', help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
        sam_parse.add_argument("--points_per_side", type=int, default=64, help="Generate masks by sampling a grid over the image with this many points to a side.")
        sam_parse.add_argument("--pred_iou_thresh", type=float, default=0.86, help="Exclude masks with a predicted score from the model that is lower than this threshold.")
        sam_parse.add_argument("--stability_score_thresh", type=float,default=0.92, help="Exclude masks with a stability score lower than this threshold.")
        sam_parse.add_argument("--crop_n_layers", type=int, default=1, help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
            ),
        )
        sam_parse.add_argument("--crop_n_points_downscale_factor", type=int, default=2, help="The number of points-per-side in each layer of crop is reduced by this factor.")
        sam_parse.add_argument("--min_mask_region_area", type=int, default=100, help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
            ),
        )
        
        sam_parse.add_argument("--convert_to_rle", action="store_true", help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
            ),
        )
        

        slam_parser = self.add_argument_group('salm')
        slam_parser.add(
            '--name', type=str,
            help='Name of the config for the offline reconstruction system.')
        slam_parser.add(
            '--fragment_size', type=int,
            help='Number of RGBD frames to construct a fragment.')
        slam_parser.add(
            '--device', type=str,
            help='Device to run the system.')
        slam_parser.add(
            '--engine', type=str,
            choices=['tensor', 'legacy'],
            help='Open3D engine to reconstruct.')
        slam_parser.add(
            '--multiprocessing', action='store_true',
            help='Use multiprocessing in operations. Only available for the legacy engine.')
        slam_parser.add(
            '--path_dataset', type=str, default='data/LoungeRGBDImages/', 
            help='Path to the dataset folder. It should contain a folder with depth and a folder with color images.')
        slam_parser.add(
            '--path_trajectory', type=str, default='data/LoungeRGBDImages/lounge_trajectory.log', 
            help='Path to the dataset folder. It should contain a folder with depth and a folder with color images.')
        slam_parser.add(
            '--depth_folder', type=str, default='depth', 
            help='Path that stores depth images.')
        slam_parser.add(
            '--color_folder', type=str, default='color', 
            help='Path that stores color images.')
        slam_parser.add(
            '--path_intrinsic', type=str,
            help='Path to the intrinsic.json config file.'
            'If the intrinsic matrix for color image is different,'
            'specify it by --path_color_intrinsic.'
            'By default PrimeSense intrinsics is used.')
        slam_parser.add(
            '--path_color_intrinsic', type=str,
            help='Path to the intrinsic.json config file.'
            'If the intrinsic matrix for color image is different,'
            'specify it by --path_color_intrinsic.'
            'By default PrimeSense intrinsics is used.')
        slam_parser.add(
            '--depth_min', type=float,
            help='Min clipping distance (in meter) for input depth data.')
        slam_parser.add(
            '--depth_max', type=float,
            help='Max clipping distance (in meter) for input depth data.')
        slam_parser.add(
            '--depth_scale', type=float,
            help='Scale factor to convert raw input depth data to meters.')

        odometry_parser = self.add_argument_group('odometry')
        odometry_parser.add(
            '--odometry_method', type=str,
            choices=['point2plane', 'intensity', 'hybrid', 'frame2model'],
            help='Method used in pose estimation between RGBD images.'
            'Frame2model only available for the tensor engine.')
        odometry_parser.add(
            '--odometry_loop_interval', type=int,
            help='Intervals to check loop closures between RGBD images.')
        odometry_parser.add(
            '--odometry_loop_weight', type=float,
            help='Weight of loop closure edges when optimizing pose graphs for odometry.')
        odometry_parser.add(
            '--odometry_distance_thr', type=float,
            help='Default distance threshold to filter outliers in odometry correspondences.')

        registration_parser = self.add_argument_group('registration')
        registration_parser.add(
            '--icp_method', type=str,
            choices=['colored', 'point2point', 'point2plane', 'generalized'],
            help='Method used in registration between fragment point clouds with a good initial pose estimate.'
            'Generalized ICP only available for the tensor engine.')
        registration_parser.add(
            '--icp_voxelsize', type=float,
            help='Voxel size used to down sample point cloud for fast/multiscale ICP.')
        registration_parser.add(
            '--icp_distance_thr', type=float,
            help='Default distance threshold to filter outliers in ICP correspondences.')
        registration_parser.add(
            '--global_registration_method', type=str,
            choices=['fgr', 'ransac'],
            help='Method used in global registration of two fragment point clouds without an initial pose estimate.')
        registration_parser.add(
            '--registration_loop_weight', type=float,
            help='Weight of loop closure edges when optimizing pose graphs for registration.')

        integration_parser = self.add_argument_group('integration')
        integration_parser.add(
            '--integration_mode',type=str,
            choices=['color', 'depth'],
            help='Volumetric integration mode.')
        integration_parser.add(
            '--voxel_size', type=float,
            help='Voxel size in meter for volumetric integration.')
        integration_parser.add(
            '--trunc_voxel_multiplier', type=float,
            help='Truncation distance multiplier in voxel size for signed distance. For instance, --trunc_voxel_multiplier=8 with --voxel_size=0.006(m) creates a truncation distance of 0.048(m).')
        integration_parser.add(
            '--est_point_count', type=int,
            help='Estimated point cloud size for surface extraction.')
        integration_parser.add(
            '--block_count', type=int,
            help='Pre-allocated voxel block count for volumetric integration.')
        integration_parser.add(
            '--surface_weight_thr', type=float,
            help='Weight threshold to filter outliers during volumetric surface reconstruction.')
        integration_parser.add(
            '--clipping_distance_in_meters', type=float,
            help='threshold to filter outliers in terms of distance.')
        # yapf:enable

    def get_config(self):
        config = self.parse_args()

        # Resolve conflicts
        if config.engine == 'legacy':
            if config.device.lower().startswith('cuda'):
                print('Legacy engine only supports CPU.', 'Fallback to CPU.')
                config.device = 'CPU:0'

            if config.odometry_method == 'frame2model':
                print('Legacy engine does not supports frame2model tracking.',
                      'Fallback to hybrid odometry.')
                config.odometry_method = 'hybrid'

        if config.engine == 'tensor':
            if config.icp_method == 'generalized':
                print('Tensor engine does not support generalized ICP.',
                      'Fallback to colored ICP.')
                config.icp_method = 'colored'

            if config.multiprocessing:
                print('Tensor engine does not support multiprocessing.',
                      'Disabled.')
                config.multiprocessing = False

        return config


if __name__ == '__main__':
    # Priority: command line > custom config file > default config file
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()
    print(config)
