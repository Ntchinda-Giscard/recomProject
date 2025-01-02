from zenml import pipeline
from mlProject.pipeline.stage_01_data_ingestion import data_ingestion
from mlProject.pipeline.stage_02_data_validation import data_validation
from mlProject.pipeline.stage_03_data_transformation import data_transformation
from mlProject.pipeline.stage_04_model_trainer import model_trainer
from mlProject.pipeline.stage_05_model_evaluation import model_evaluation


@pipeline
def training_pipeline():
    data_ingestion()
    data_validation()
    data_transformation()
    model_trainer()
    model_evaluation()


if __name__ == "__main__":
    training_pipeline()