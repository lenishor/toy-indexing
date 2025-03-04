import torch

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfig:
    num_symbols: int = MISSING
    array_len: int = MISSING
    num_steps: int = MISSING
    batch_size: int = 1

@dataclass
class ModelConfig:
    embedding_dim: int = MISSING

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta_1: float = 0.9
    beta_2: float = 0.999
    grad_clip: float = 1.0

@dataclass
class SchedulerConfig:
    type: str = "cosine"  # "cosine" or "constant"
    min_learning_rate: float = 0.0

@dataclass
class EvalConfig:
    num_samples: int = 1_000

@dataclass
class LogConfig:
    log_every: int = 1
    save_every: int = 1
    eval_every: int = 1

@dataclass
class RunConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: Optional[str] = None
    project: str = "toy-indexing-task"
    entity: str = "physicsofintelligence"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

def setup_config_store():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RunConfig)
