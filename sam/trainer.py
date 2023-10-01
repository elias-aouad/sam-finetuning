from functools import cached_property
from pydantic import BaseModel

import torch
import torch.nn as nn

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import load_from_safetensors
from refiners.training_utils.trainer import Trainer
from refiners.foundationals.segment_anything.image_encoder import SAMViTH
from refiners.foundationals.segment_anything.mask_decoder import MaskDecoder
from refiners.foundationals.segment_anything.prompt_encoder import MaskEncoder, PointEncoder
from refiners.foundationals.segment_anything.model import SegmentAnything
from refiners.training_utils.config import BaseConfig

from .sam_dataset import SamDataset


SAM_WEIGHTS_PATH: str = ""
IMAGE_DIR: str = ""
MASK_DIR: str = ""


class SafetensorsConfig(BaseModel):
    path: str = SAM_WEIGHTS_PATH


class DatasetConfig(BaseModel):
    root_dir: str = IMAGE_DIR
    mask_dir: str = MASK_DIR


class SegmentAnythingConfig(BaseConfig):
      safetensors: SafetensorsConfig = SafetensorsConfig()
      dataset: DatasetConfig = DatasetConfig()


class SegmentAnythingTrainer(Trainer):

    @cached_property
    def state_dict(self):
      state_dict = load_from_safetensors(self.config.safetensors.path)
      sam_parts = set([weight_name.split(".")[0] for weight_name in state_dict.keys()])
      partitioned_dict = {
          part: {
              weight_name: weight
              for weight_name, weight in state_dict.items()
              if weight_name.split(".")[0] == part
          }
          for part in sam_parts
      }
      return partitioned_dict

    @cached_property
    def image_encoder(self) -> SAMViTH:
        same_vith = SAMViTH().to(device=self.device)
        same_vith.load_state_dict(self.state_dict["image_encoder"], strict=False)
        return same_vith

    @cached_property
    def point_encoder(self) -> PointEncoder:
        point_encoder = PointEncoder().to(device=self.device)
        point_encoder.load_state_dict(self.state_dict["point_encoder"], strict=False)
        return point_encoder

    @cached_property
    def mask_encoder(self) -> MaskEncoder:
        mask_encoder = MaskEncoder().to(device=self.device)
        mask_encoder.load_state_dict(self.state_dict["mask_encoder"], strict=False)
        return mask_encoder

    @cached_property
    def mask_decoder(self) -> MaskDecoder:
        mask_decoder = MaskEncoder().to(device=self.device)
        mask_decoder.load_state_dict(self.state_dict["mask_decoder"], strict=False)
        return mask_decoder

    @cached_property
    def sam(self) -> SegmentAnything:
        sam = SegmentAnything(
            image_encoder=self.image_encoder,
            point_encoder=self.point_encoder,
            mask_encoder=self.mask_encoder,
            mask_decoder=self.mask_decoder
        )
        return sam

    def load_models(self) -> dict[str, fl.Module]:
        return {"sam": self.sam}

    def load_dataset(self) -> SamDataset:
        return SamDataset(
            root_dir=self.config.dataset.root_dir,
            mask_dir=self.config.dataset.mask_dir
        )

    def compute_loss(self, batch):
        image, foreground_point, mask = batch
        output_mask, _, _ = self.sam.predict(
            input=image, foreground_points=foreground_point
        )
        output_mask_1, output_mask_2, output_mask_3 = (
            output_mask[0, 0], output_mask[0, 1], output_mask[0, 3]
        )
        mask = torch.from_numpy(mask).flatten()

        loss = nn.BCELoss(mask, output_mask_1)
        loss += nn.BCELoss(mask, output_mask_2)
        loss += nn.BCELoss(mask, output_mask_3)
        return loss
