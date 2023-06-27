import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from config import ConfigParser

import os, sys
import numpy as np
import threading
import time
from common import load_rgbd_file_names, save_poses, load_intrinsic, extract_trianglemesh


class Reconstruction:

    def __init__(self, config):
        self.config = config
        self.is_done = False
        self.flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        self.idx = 0
        self.poses = []
        self._on_start()

        # Start running
        threading.Thread(name='UpdateMain', target=self.update_main).start()


    # On start: point cloud buffer and model initialization.
    def _on_start(self):
        
        device = o3d.core.Device(self.config.device)
        max_points = self.config.est_point_count
        pcd_placeholder = o3d.t.geometry.PointCloud(device)

        pcd_placeholder.point["positions"] = o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32))
        pcd_placeholder.point["colors"] =  o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32))


        T_frame_to_model = o3c.Tensor(np.identity(4))
        
        self.model = o3d.t.pipelines.slam.Model(
            self.config.voxel_size, 16,
            self.config.block_count, T_frame_to_model,
            o3c.Device(self.config.device))
        self.is_started = True

    def _on_close(self):
        self.is_done = True

        if self.is_started:
            
            print('Saving model to {}...'.format(self.config.output_name))
            self.model.voxel_grid.save(os.path.join(self.config.path_dataset, self.config.output_name))
            print('Finished.')

            mesh_fname = os.path.join(self.config.path_dataset, ('.'.join(self.config.output_name.split('.')[:-1]) + '.ply'))
            print('Extracting and saving mesh to {}...'.format(mesh_fname))
            mesh = extract_trianglemesh(self.model.voxel_grid, self.config,
                                        mesh_fname)
            print('Finished.')
            
            pc_fname = os.path.join(self.config.path_dataset, ('.'.join(self.config.output_name.split('.')[:-1]) + '.pcd'))

            if self.config.engine == 'legacy':
                pc = self.model.extract_pointcloud().transform(self.flip_transform)
            else:
                pc = self.model.extract_pointcloud().transform(self.flip_transform).to_legacy()
            print('Extracting and saving point cloud to {}...'.format(pc_fname))
            o3d.io.write_point_cloud(pc_fname, pc)
            print('Finished.')  

            log_fname = os.path.join(self.config.path_dataset, ('.'.join(self.config.output_name.split('.')[:-1]) + '.log'))
            print('Saving trajectory to {}...'.format(log_fname))
            save_poses(log_fname, self.poses)
            print('Finished.')

        return True

    # Major loop
    def update_main(self):
        depth_file_names, color_file_names = load_rgbd_file_names(self.config)
        #print(depth_file_names, color_file_names)

        
        intrinsic = load_intrinsic(self.config)
        
        #print(len(depth_file_names), len(color_file_names))

        n_files = len(color_file_names)
        device = o3d.core.Device(self.config.device)

        T_frame_to_model = o3c.Tensor(np.identity(4))
        depth_ref = o3d.t.io.read_image(depth_file_names[0])
        color_ref = o3d.t.io.read_image(color_file_names[0])
        input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                 depth_ref.columns, intrinsic,
                                                 device)
        raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                                   depth_ref.columns, intrinsic,
                                                   device)

        input_frame.set_data_from_image('depth', depth_ref)
        np.array(depth_ref)
        input_frame.set_data_from_image('color', color_ref)
        np.array(color_ref)

        raycast_frame.set_data_from_image('depth', depth_ref)
        raycast_frame.set_data_from_image('color', color_ref)


        fps_interval_len = 30
        self.idx = 0
        pcd = None

        start = time.time()
        while not self.is_done:

            depth = o3d.t.io.read_image(depth_file_names[self.idx]).to(device)
            color = o3d.t.io.read_image(color_file_names[self.idx]).to(device)

            input_frame.set_data_from_image('depth', depth)
            input_frame.set_data_from_image('color', color)
            
            try:

                if self.idx > 0:
                    result = self.model.track_frame_to_model(
                        input_frame,
                        raycast_frame,
                        float(self.config.depth_scale),
                        self.config.depth_max,
                    )
                    T_frame_to_model = T_frame_to_model @ result.transformation

                self.poses.append(T_frame_to_model.cpu().numpy())
                self.model.update_frame_pose(self.idx, T_frame_to_model)
                self.model.integrate(input_frame,
                                    float(self.config.depth_scale),
                                    self.config.depth_max,
                                    self.config.trunc_voxel_multiplier)
                self.model.synthesize_model_frame(
                    raycast_frame, float(self.config.depth_scale),
                    self.config.depth_min, self.config.depth_max,
                    self.config.trunc_voxel_multiplier,
                    True)
                
                # Output FPS
                if (self.idx % fps_interval_len == 0):
                    end = time.time()
                    elapsed = end - start
                    start = time.time()
                    print('FPS: {:.3f}'.format(fps_interval_len / elapsed))

                # Output info
                info = 'Frame {}/{}\n\n'.format(self.idx, n_files)
                info += 'Transformation:\n{}\n'.format(
                    np.array2string(T_frame_to_model.numpy(),
                                    precision=3,
                                    max_line_width=40,
                                    suppress_small=True))
                info += 'Active voxel blocks: {}/{}\n'.format(
                    self.model.voxel_grid.hashmap().size(),
                    self.model.voxel_grid.hashmap().capacity())
                info += 'Surface points: {}/{}\n'.format(
                    0 if pcd is None else pcd.point['positions'].shape[0],
                    self.config.est_point_count)
                print(info)
            except:
                print("Unexpected error: ", sys.exc_info()[1])
            self.idx += 1
            self.is_done = self.is_done | (self.idx >= n_files)
        
        self._on_close()
        
def run_slam(config):
    #print(config)
    w = Reconstruction(config)
