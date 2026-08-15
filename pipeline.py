from src.configs import DataParameters, ModelConfig, TrainingArguments
from src.models.create_model import create_model
from src.data.pipeline import DataPipeline
from src.training.trainer import Trainer


class Pipeline:
    def __init__(self, task:str, model, data_params:DataParameters, model_config:ModelConfig, train_args:TrainingArguments, optimiser, plot_loss=False):
        # DEFAULT CONFIGURATIONS
        self.task = task

        self.data_params = data_params
        self.model_config = model_config
        self.train_args = train_args

        if type(model) == str:
            self.model = create_model(model, self.model_config)
        else:
            self.model = model

        self.optimiser = optimiser

        self.trainer = Trainer(
                    data_pipeline=DataPipeline(self.data_params,
                                               self.train_args,
                                               self.model_config),
                    model=self.model,
                    train_args=self.train_args,
                    optimiser=self.optimiser,)

        self.plot_loss = plot_loss


    def __call__(self):
        if self.task == 'train':
            return self.trainer.train(self.plot_loss)
        elif self.task == 'evaluate':
            return self.trainer.evaluate(self.plot_loss)
        elif self.task == 'train_evaluate':
            return self.trainer.train_evaluate(self.plot_loss)
