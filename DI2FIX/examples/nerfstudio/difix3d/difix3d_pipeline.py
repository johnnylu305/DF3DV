# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field
from typing import Optional, Type
from pathlib import Path
from PIL import Image
import os
import tqdm
import random
import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs


from difix3d.difix3d_datamanager import (
    Difix3DDataManagerConfig,
)
from src.pipeline_difix import DifixPipeline
from examples.utils import CameraPoseInterpolator


@dataclass
class Difix3DPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: Difix3DPipeline)
    """target class to instantiate"""
    datamanager: Difix3DDataManagerConfig = Difix3DDataManagerConfig()
    """specifies the datamanager config"""
    steps_per_fix: int = 2000
    """rate at which to fix artifacts"""
    steps_per_val: int = 5000
    """rate at which to evaluate the model"""

class Difix3DPipeline(VanillaPipeline):
    """Difix3D pipeline"""

    config: Difix3DPipelineConfig

    def __init__(
        self,
        config: Difix3DPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        render_dir: str = "renders",
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.render_dir = render_dir
        
        self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        self.difix.to("cuda")

        self.training_poses = self.datamanager.train_dataparser_outputs.cameras.camera_to_worlds
        self.training_poses = torch.cat([self.training_poses, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.training_poses.shape[0], 1, 1)], dim=1)
        self.testing_poses = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split).cameras.camera_to_worlds
        self.testing_poses = torch.cat([self.testing_poses, torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.testing_poses.shape[0], 1, 1)], dim=1)
        self.current_novel_poses = self.training_poses
        self.current_novel_cameras = self.datamanager.train_dataparser_outputs.cameras

        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
        self.novel_datamanagers = []

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if len(self.novel_datamanagers) == 0 or random.random() < 0.6:
            ray_bundle, batch = self.datamanager.next_train(step)
        else:
            ray_bundle, batch = self.novel_datamanagers[-1].next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # run fixer
        if (step % self.config.steps_per_fix == 0):
            self.fix(step)

        # run evaluation
        if (step % self.config.steps_per_val == 0):
            self.val(step)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

    @torch.no_grad()
    def render_traj(self, step, cameras, tag="novel"):
        for i in tqdm.trange(0, len(cameras), desc="Rendering trajectory"):
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(cameras[i])

            rgb_path = f"{self.render_dir}/{tag}/{step}/Pred/{i:04d}.png"
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            rgb_canvas = outputs['rgb'].cpu().numpy()
            rgb_canvas = (rgb_canvas * 255).astype(np.uint8)
            Image.fromarray(rgb_canvas).save(rgb_path)

    @torch.no_grad()
    def val(self, step):
        cameras = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split).cameras
        for i in tqdm.trange(0, len(cameras), desc="Running evaluation"):
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(cameras[i])

            rgb_path = f"{self.render_dir}/val/{step}/{i:04d}.png"
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            rgb_canvas = outputs['rgb'].cpu().numpy()
            rgb_canvas = (rgb_canvas * 255).astype(np.uint8)
            Image.fromarray(rgb_canvas).save(rgb_path)

    @torch.no_grad()
    def fix(self, step: int):

        novel_poses = self.interpolator.shift_poses(self.current_novel_poses.numpy(), self.testing_poses.numpy(), distance=0.5)
        novel_poses = torch.from_numpy(novel_poses).to(self.testing_poses.dtype)

        ref_image_indices = self.interpolator.find_nearest_assignments(self.training_poses.numpy(), novel_poses.numpy())
        ref_image_filenames = np.array(self.datamanager.train_dataparser_outputs.image_filenames)[ref_image_indices].tolist()

        cameras = self.datamanager.train_dataparser_outputs.cameras
        cameras = Cameras(
            fx=cameras.fx[0].repeat(len(novel_poses), 1),
            fy=cameras.fy[0].repeat(len(novel_poses), 1),
            cx=cameras.cx[0].repeat(len(novel_poses), 1),
            cy=cameras.cy[0].repeat(len(novel_poses), 1),
            distortion_params=cameras.distortion_params[0].repeat(len(novel_poses), 1),
            height=cameras.height[0].repeat(len(novel_poses), 1),
            width=cameras.width[0].repeat(len(novel_poses), 1),
            camera_to_worlds=novel_poses[:, :3, :4],
            camera_type=cameras.camera_type[0].repeat(len(novel_poses), 1),
            metadata=cameras.metadata,
        )

        self.render_traj(step, cameras)

        image_filenames = []
        for i in tqdm.trange(0, len(novel_poses), desc="Fixing artifacts..."):
            image = Image.open(f"{self.render_dir}/novel/{step}/Pred/{i:04d}.png").convert("RGB")
            ref_image = Image.open(ref_image_filenames[i]).convert("RGB")
            output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            output_image = output_image.resize(image.size, Image.LANCZOS)
            os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
            output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")
            image_filenames.append(Path(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png"))
            if ref_image is not None:
                os.makedirs(f"{self.render_dir}/novel/{step}/Ref", exist_ok=True)
                ref_image.save(f"{self.render_dir}/novel/{step}/Ref/{i:04d}.png")

        dataparser_outputs = self.datamanager.train_dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=dataparser_outputs.scene_box,
            mask_filenames=None,
            dataparser_scale=dataparser_outputs.dataparser_scale,
            dataparser_transform=dataparser_outputs.dataparser_transform,
            metadata=dataparser_outputs.metadata,
        )

        datamanager_config = Difix3DDataManagerConfig(
            dataparser=self.config.datamanager.dataparser,
            train_num_rays_per_batch=16384,
            eval_num_rays_per_batch=4096,
        )
        
        datamanager = datamanager_config.setup(
            device=self.datamanager.device, 
            test_mode=self.datamanager.test_mode, 
            world_size=self.datamanager.world_size, 
            local_rank=self.datamanager.local_rank
        )

        datamanager.train_dataparser_outputs = dataparser_outputs
        datamanager.train_dataset = datamanager.create_train_dataset()
        datamanager.setup_train()

        self.novel_datamanagers.append(datamanager)
        self.current_novel_poses = novel_poses
        self.current_novel_cameras = cameras