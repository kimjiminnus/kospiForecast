# Transformer Encoder vs TimesFM 2.5 for Multivariate Financial Time-Series Forecasting
A PyTorch project investigating the forecasting performance of a custom Transformer Encoder-only architecture against Google's TimesFM 2.5 Decoder-only model used as a zero-shot baseline.

(By someone forced into a long-term investor by the KOSPI😔)


## Overview
The project provides a reusable, modular framework for training and evaluating Transformer-based models on multivariate financial time-series data.

## Configurations
Dataclasses for simpler parameter inputs.

```python
class DataParameters:
    target:str = 'target/^KS11'     # Format of Target Index: target/ticker
    var1:str = "krw/USDKRW=X".      # Format of Variable Indices: desired_name/ticker
    var2:str = "dxy/DX-Y.NYB"
    var3:str = "sox/^SOX"

    start_date:str = "2023-06-01".  # Start date of data retrieval
    end_date:str = "2026-07-01".    # End date of data retrieval

    test_split:float = 0.3

    inference:bool = False


class ModelConfig:
    d_model:int = 8       # Feature Dimensions
    dim_ffn:int = 16.     # Feed-forward network dimension
    max_window:int = 5    # Maximum sequence window
    num_heads:int = 2.    # For Multi-head Attention
    num_layers:int = 3.   # Number of Encoder Layers
    num_vars:int = 3      # Fixed due to high correlation of market indices
    dropout:float = 0.2   # Prevents overfitting on small dataset


class TrainingArguments:
    epochs: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-2  
    weight_decay: float = 1e-4
    criterion :str = 'mse'       # Alternatively: 'mae', 'huber'
```


Training Arguments 
- Epochs
- Train & Eval batch size
- Learning Rate
- Weight Decay
- Criterion (e.g. MSE, MAE)

## DataPipeline
The DataPipeline class covers the following data preparation workflow in order with a simple .run()
 - Market data retrieval
 - Temporal alignment
 - Log-return transformation
 - Instantiation of custom TimeSeries Dataset using data from previous preprocessing steps
 - Train-test split
 - Standard feature scaling
 - Returning train and validation dataloaders
```python
# Instantiate a DataPipeline
data_pipeline = DataPipeline(
                      data_params=DataParameters(),
                      train_args=TrainingArguments(),
                      model_config=ModelConfig()
                      )

# Return Dataloaders
train_loader, val_loader = data_pipeline.run()       
```

## Model Creation
```python
# Creates, loads, and returns a Transformer Encoder model
create_model("encoder", model_config)

# Does the same for a TimesFm 2.5 model
create_model("timesfm", model_config=None)
```

## Custom Transformer Encoder
- Converts stock data into numerical feature embeddings. i.e, If k market indexes are used to predict a target index, shape of input tensor will be (batch_size, max_window, k)
- Applies positional encoding to provide temporal information from previous trading days to the model.

## Google's TimesFM 2.5 Model
This Decoder-only model is used strictly for zero-shot forecasting as a baseline model for comparison with the performance of a custom Transformer Encoder-only model

## Trainer Class
- Trainer simplifies tasks such as training only, evaluation only, or both at the same time.
```python
# Instantiate Trainer
trainer = Trainer(
            data_pipeline=DataPipeline(),
            model=model,
            train_args=TrainingArguments,
            optimiser=optim.Adam
            )

# Example: Train only, returns a list of training loss for every epoch
trainer.train(plot_loss=False)

# Example: Train and evaluate, returns a plot with both loss curves, and their respective list of losses
trainer.train_evaluate(plot_loss=True) 
```
## Pipeline Class
- Pipeline covers instantiation of both DataPipeline & Trainer classes and combines the process.
```python
pipeline_object = Pipeline(
           task="train",
           model="encoder"
           model_config=ModelConfig(),
           train_args=TrainingArguments(),
           optimiser=optim.SGD
           plot_loss=True
           )
```
- Users can input pre-existing models for fine-tuning instead of string inputs
```python
# This line is all that's required to train or evaluate a certain configuration or data!
pipeline_object()
```

## HyperparamTuner Class
- HyperparamTuner provides Optuna lists of possible hyperpameter values to automate hyperparameter optimisation.
- Instantiates Pipeline within its own method, allowing users to experiment multiple Pipeline objects easily

## get_optimal_hyperparams() function
- Performs data preprocessing, model creation, model training and hyperparameter optimisation all with one line of code😎😎
```python
best_trial, best_params = get_optimal_hyperparams(
                              epochs=10,
                              model_type="encoder",
                              data_params=DataParameters(),
                              optimiser=optim.AdamW,
                              n_trials=20.    # No. of combinations of hyperparams to try out
                              plot_loss=False
                              )
```                     





