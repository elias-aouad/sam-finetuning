from .trainer import SegmentAnythingConfig, SegmentAnythingTrainer
from .callbacks import LoadLoras, TurnOffGradients


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = SegmentAnythingConfig.load_from_toml(
        toml_path=config_path,
    )
    callbacks = [LoadLoras(), TurnOffGradients()]

    trainer = SegmentAnythingTrainer(config=config, callbacks=callbacks)
    trainer.train()
