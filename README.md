# Transformer Encoder vs TimesFM 2.5 for Multivariate Financial Time-Series Forecasting
A PyTorch project investigating the forecasting performance of a custom Transformer Encoder-only architecture against Google's TimesFM 2.5 Decoder-only model used as a zero-shot baseline.

(By someone forced into a long-term investor by the KOSPI😔)


## Overview
The project uses reusable DataParameters, ModelConfigs and Training Arguments data classes for simplified parameter inputs.

The DataPipeline covers market data retrieval, temporal alignment, log return transformation, data preprocessing, feature scaling, instantiation of a custom TimeSeriesDataset class and returning them in the form of train and val dataloaders.

The create_model is called with parameters ModelConfigs and a string i.e "encoder", "timesfm" that instantiates, loads and returns models accordingly.

The Custom Transformer Encoder Model applies numerical feature embeddings and sinusoidal positional encodings onto input matrices before processing, while the TimesFM 2.5 model is used strictly for zero-shot forecasting.

The Trainer Class accepts a DataPipeline, model, TrainingArguments and an optimiser. It can be called for different purposes such as training only, evaluation only, and both. plot_loss() can be used to plot losses when a method is called.

The Pipeline Class covers the whole process and instantiates the DataPipeline and the Trainer. It can accept both pre-existing models for fine-tuning purposes, or strings i.e "encoder" for on-the-spot instantiation and training.

HyperparamTuner provides lists of possible hyperparameter values for Optuna to automate hyperparameter optimisation. While Pipeline can only be used to train and evaluate on a certain configuration, HyperParamTuner instantiates Pipeline concurrently, therefore allowing users to complete data preprocessing, training and obtaining optimal hyperparameters with a single get_optimal_hyperparams() function.



