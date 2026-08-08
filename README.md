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
The DataPipeline class covers the following data preparation workflow in order
 - Market data retrieval
 - Temporal alignment
 - Log-return transformation
 - Instantiation of custom TimeSeries Dataset using data from previous preprocessing steps
 - Standard feature scaling
 - Train-test split
 - Returning train and validation dataloaders

## Model Creation
create_model("encoder", ModelConfigs) creates a Transformer Encoder model </br>
create_model("timesfm, ModelConfigs)


The create_model is called with parameters ModelConfigs and a string i.e "encoder", "timesfm" that instantiates, loads and returns models accordingly.

The Custom Transformer Encoder Model applies numerical feature embeddings and sinusoidal positional encodings onto input matrices before processing, while the TimesFM 2.5 model is used strictly for zero-shot forecasting.

The Trainer Class accepts a DataPipeline, model, TrainingArguments and an optimiser. It can be called for different purposes such as training only, evaluation only, and both. plot_loss() can be used to plot losses when a method is called.

The Pipeline Class covers the whole process and instantiates the DataPipeline and the Trainer. It can accept both pre-existing models for fine-tuning purposes, or strings i.e "encoder" for on-the-spot instantiation and training.

HyperparamTuner provides lists of possible hyperparameter values for Optuna to automate hyperparameter optimisation. While Pipeline can only be used to train and evaluate on a certain configuration, HyperParamTuner instantiates Pipeline concurrently, therefore allowing users to complete data preprocessing, training and obtaining optimal hyperparameters with a single get_optimal_hyperparams() function.



