from dataclasses import MISSING, asdict, dataclass
from datetime import datetime
from typing import Optional

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}')


@dataclass
class ModelConfig:
    input_channels: int = 2
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10


@dataclass
class DataConfig:
    root_dir: str = MISSING
    batch_size: int = 16
    num_workers: int = 4
    train_fraction: float = 0.8


@dataclass
class DistributedConfig:
    distributed: bool = False
    world_size: int = 2


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False

    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100


@dataclass
class Config:
    model_dir: str = MISSING

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig(root_dir="")
    distributed: DistributedConfig = DistributedConfig()
    trainer: TrainerConfig = TrainerConfig()


def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    base_cfg = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg


if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(asdict(config))
