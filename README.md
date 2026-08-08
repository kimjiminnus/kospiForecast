# Transformer Encoder vs TimesFM 2.5 for Multivariate Financial Time-Series Forecasting
A PyTorch project investigating the forecasting performance of a custom Transformer Encoder-only architecture against Google's TimesFM 2.5 Decoder-only model used as a zero-shot baseline.

(By someone forced into a long-term investor by the KOSPI😔)


## Overview
The project provides a reusable, modular framework for training and evaluating Transformer-based models on multivariate financial time-series data.

## Configurations
Dataclasses for simpler parameter inputs.

DataParameters
- Target stock index
- Variable stock indices
- Start & end date of retrieval

ModelConfigs 
- Model dimensions
- Feedforward Network dimensions
- Max Window (seq_length for LLMs)
- Number of heads, encoder layers
- Dropout 

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

## Model Creation
```python
# Creates, loads, and returns a Transformer Encoder model
create_model("encoder", model_configs)

# Does the same for a TimesFm 2.5 model
create_model("timesfm", model_configs=None)
```

## Custom Transformer Encoder
- Converts stock data into numerical feature embeddings. i.e, If k market indexes are used to predict a target index, shape of input tensor will be (batch_size, max_window, k)
- Applies positional encoding to provide temporal information from previous trading days to the model.

## Google's TimesFM 2.5 Model
This Decoder-only model is used strictly for zero-shot forecasting as a baseline model for comparison with the performance of a custom Transformer Encoder-only model

## Trainer Class
- Trainer simplifies tasks such as training only, evaluation only, or both at the same time.
```python
# Instantiate
trainer = Trainer(
            data_pipeline=DataPipeline(),
            model=model,
            train_args=TrainingArguments,
            optimiser=optimiser
            )

# Example: Train only, returns a list of training loss for every epoch
trainer.train(plot_loss=False)

# Example: Train and evaluate, returns a plot with both loss curves, and their respective list of losses
trainer.train_evaluate(plot_loss=True) 
```
## Pipeline Class
- Pipeline covers instantiation of both DataPipeline and Trainer classes and combined the process. For example:
```python
pipeline_object = Pipeline(
           task="train",
           model="encoder"
           model_config=ModelConfig(),
           train_args=TrainingArguments(),
           optimiser=optimiser,
           plot_loss=True
              )
```
- Users can input pre-existing models for fine-tuning instead of inputting a string
```python
# This line is all you need to train or evaluate on a certain configuration!
pipeline_object()
```

## HyperparamTuner Class
- HyperparamTuner provides Optuna the lists of possible hyperpameter values to automate hyperparameter optimisation.
- Instantiates Pipeline within its own method, allowing users to experiment multiple Pipeline objects easily

## get_optimal_hyperparams() function
- Performs data preprocessing, model creation & training and hyperparameter optimisation all with one line of code😎😎
- i.e. best_trial, best_params = get_optimal_hyperparams(epochs, other parameters)





