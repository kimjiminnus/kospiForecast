from src.data.preprocessor import DataPreprocessor
from src.data.time_series_dataset import TimeSeriesDataset

from transformers import TimesFm2_5ModelForPrediction
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader


class ZeroShotTimesFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers")

    def eval_classification(self, test_loader):
        correct = 0
        for _, y in test_loader:
            y = y.to(device)
            y_pred = model(y)
            bool_tensor = y_pred * y >= 0
            correct_predictions = bool_tensor.float().sum().item()

            return f"{correct_predictions}/{y.size(0)} predicted with same direction."


    def eval_regression(self, test_loader, eval_metric, device):
        test_loss = 0.

        if eval_metric == 'mse':
            criterion = nn.MSELoss()
        elif eval_metric == 'mae':
            criterion = nn.L1Loss()
        elif eval_metric == 'huber':
            criterion = nn.HuberLoss()
        else:
            raise TypeError(f'Evaluation metric {eval_metric} is not supported.')

        for _, y in test_loader:
            y = y.to(device)
            y_pred = self.model(y)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
        return test_loss

    def evaluate(self, data_params, eval_metric:str):

        preprocessor = DataPreprocessor(data_params)
        df = preprocessor.pre_split_preprocess(
                                preprocessor.retrieve_data()
                                )
        array = np.array(df)
        dataset = TimeSeriesDataset(array, model.config.max_window)

        test_loader = DataLoader(dataset, batch_size=len(dataset.num_windows), shuffle=False)

        if eval_metric == 'classification':
            return self.eval_classification(test_loader)
        elif eval_metric == 'mse' or eval_metric == 'mae' or eval_metric == 'huber':
            return self.eval_regression(test_loader, eval_metric)
        else:
            raise TypeError(f'Evaluation metric {eval_metric} is not supported.')


    def forward(self, x:Tensor):
        forecast = self.model(x).mean_predictions
        return forecast[:,:1]
