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
 - Standard feature scaling
 - Train-test split
 - Returning train and validation dataloaders

## Model Creation
create_model("encoder", ModelConfigs):  Creates , loads, and returns a Transformer Encoder model </br>
create_model("timesfm, ModelConfigs=None):  Does the same for a ZeroShotTimesFM model

## Custom Transformer Encoder
- Converts stock data into numerical feature embeddings. i.e, If k market indexes are used to predict a target index, shape of input tensor will be (batch_size, max_window, k)
- Applies positional encoding to learn patterns of noises from previous trading days.

## Google's TimesFM 2.5 Model
This model is used strictly for zero-shot forecasting to get an unbiased evaluation of the performance of a custom Transformer Encoder-only model against a pretrained Decoder-only model.

## Trainer Class
- Accepts DataPipeline, model, TrainingArguments and optimiser as input parameters
- Can be used for training only, evaluation only, or both.
- plot_loss=True can be input to return a loss graph (plot_loss=False by default)
- i.e. trainer.train_evaluate(plot_loss=True) returns a single plot with train and validation loss curves

## Pipeline Class
- Pipeline covers instantiation of both DataPipeline and Trainer classes
- Users can input pre-existing models for fine-tuning, or strings i.e "encoder" for on-the-spot instantiation and training
- pipeline_object() is all that's required to train and evaluate a certain configuration or data!

## HyperparamTuner Class
- HyperparamTuner provides Optuna the lists of possible hyperpameter values to automate hyperparameter optimisation.
- Instantiates Pipeline within its own method, allowing users to experiment multiple Pipeline objects easily

## get_optimal_hyperparams() function
- Performs data preprocessing, model creation & training and hyperparameter optimisation all with one line of code😎😎
- i.e. best_trial, best_params = get_optimal_hyperparams(epochs, other parameters)





