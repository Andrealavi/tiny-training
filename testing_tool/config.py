from pydantic import BaseModel, Field
from typing import Optional, List


class DataProviderConfig(BaseModel):
    dataset: str = Field(default="image_folder", description="Dataset type")
    root: str = Field(default="/home/alavino/dataset/", description="Dataset root path")
    resize_scale: float = Field(default=0.08, ge=0.0, le=1.0, description="Resize scale factor")
    color_aug: float = Field(default=0.4, ge=0.0, description="Color augmentation factor")
    base_batch_size: int = Field(default=64, gt=0, description="Base batch size")
    n_worker: int = Field(default=8, ge=0, description="Number of workers")
    image_size: int = Field(default=128, gt=0, description="Image size")
    num_classes: int = Field(default=102, gt=0, description="Number of classes")


class RunConfig(BaseModel):
    n_epochs: int = Field(default=50, gt=0, description="Number of epochs")
    base_lr: float = Field(default=0.025, gt=0.0, description="Base learning rate")
    bs256_lr: float = Field(default=0.01, gt=0.0, description="Batch size 256 learning rate")
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")
    warmup_lr: float = Field(default=0.0, ge=0.0, description="Warmup learning rate")
    lr_schedule_name: str = Field(default="cosine", description="Learning rate schedule name")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay")
    no_wd_keys: List[str] = Field(default=["norm", "bias"], description="Keys without weight decay")
    optimizer_name: str = Field(default="sgd", description="Optimizer name")
    bias_only: bool = Field(default=False, description="Bias only training")
    fc_only: bool = Field(default=False, description="Fully connected only training")
    fc_lr10: bool = Field(default=False, description="FC layer 10x learning rate")
    eval_per_epochs: int = Field(default=10, gt=0, description="Evaluation frequency in epochs")
    grid_output: Optional[str] = Field(default=None, description="Grid output path")
    grid_ckpt_path: Optional[str] = Field(default=None, description="Grid checkpoint path")
    n_block_update: int = Field(default=-1, description="Number of block updates (-1 for all)")


class NetConfig(BaseModel):
    net_name: str = Field(default="mbv2-w0.35", description="Network name")
    pretrained: bool = Field(default=False, description="Use pretrained model")
    cls_head: str = Field(default="linear", description="Classification head type")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    mcu_head_type: str = Field(default="fp", description="MCU head type")


class BackwardConfig(BaseModel):
    enable_backward_config: bool = Field(default=False, description="Enable backward configuration")
    n_bias_update: Optional[int] = Field(default=None, ge=0, description="Number of bias updates")
    n_weight_update: Optional[int] = Field(default=None, ge=0, description="Number of weight updates")
    weight_update_ratio: Optional[List[float]] = Field(default=None, description="Weight update ratio")
    weight_select_criteria: str = Field(default="magnitude+", description="Weight selection criteria")
    pw1_weight_only: bool = Field(default=False, description="PW1 weight only")
    manual_weight_idx: Optional[List[int]] = Field(default=None, description="Manual weight indices")
    manual_bias_idx: Optional[List[int]] = Field(default=None, description="Manual bias indices")
    quantize_gradient: bool = Field(default=False, description="Quantize gradient")
    freeze_fc: bool = Field(default=False, description="Freeze fully connected layer")
    train_scale: bool = Field(default=False, description="Training scale")


class TrainingConfig(BaseModel):
    """
    Class used to validate training configuration data.

    This class will be used to check that the types of data that are passed
    to the training function are correct.

    Attributes name are taken from the original project code,
    with the exception of the weights and biases training attributes that
    were considered a bit unclear with their original naming.
    """
    run_dir: Optional[str] = Field(default="runs", description="Run directory path")
    manual_seed: int = Field(default=0, ge=0, description="Manual seed for reproducibility")
    evaluate: bool = Field(default=False, description="Whether to evaluate")
    ray_tune: bool = Field(default=False, description="Ray tune configuration")
    resume: bool = Field(default=False, description="Resume training")
    data_provider: DataProviderConfig = Field(default_factory=DataProviderConfig, description="Data provider configuration")
    run_config: RunConfig = Field(default_factory=RunConfig, description="Run configuration")
    net_config: NetConfig = Field(default_factory=NetConfig, description="Network configuration")
    backward_config: BackwardConfig = Field(default_factory=BackwardConfig, description="Backward pass configuration")

    class Config:
        """Pydantic configuration class."""
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Prevents extra fields from being added
