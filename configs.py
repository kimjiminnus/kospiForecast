#DEFAULT CONFIGS

@dataclass
class DataParameters:
    target:str = "target/^KS11"
    var1:str = "krw/USDKRW=X"
    var2:str = "dxy/DX-Y.NYB"
    var3:str = "sox/^SOX"

    start_date:str = "2023-06-01"
    end_date:str = "2026-07-01"

    test_split:float = 0.3

    inference:bool = False


@dataclass
class ModelConfig:
    d_model:int = 8
    dim_ffn:int = 16
    max_window:int = 5
    num_heads:int = 2
    num_layers:int = 3
    num_vars:int = 3
    dropout:float = 0.2


@dataclass
class TrainingArguments:
    epochs: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    criterion :str = 'mse'
