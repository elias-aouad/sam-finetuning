import refiners.fluxion.layers as fl
from refiners.training_utils.callback import Callback
from refiners.fluxion.adapters.lora import SingleLoraAdapter

from .trainer import SegmentAnythingTrainer


class LoadLoras(Callback[SegmentAnythingTrainer]):
    def on_train_begin(self, trainer: SegmentAnythingTrainer) -> None:
        lora_config = trainer.config.lora

        for layer in trainer.sam.mask_decoder.layers(fl.Attention):
            for linear, parent in layer.walk(fl.Linear):
                SingleLoraAdapter(target=linear, rank=lora_config.rank).inject(parent)


class TurnOffGradients(Callback[SegmentAnythingTrainer]):
    def on_train_begin(self, trainer: SegmentAnythingTrainer) -> None:
        for param_name, param in trainer.sam.named_parameters():
            param.requires_grad = ("SingleLoraAdapter" in param_name)
