from zenml import pipeline
from mlProject.stage.stage_01_data_ingestion import data_ingestion
from mlProject.stage.stage_02_data_validation import data_validation
from mlProject.stage.stage_03_data_transformation import data_transformation
from mlProject.stage.stage_04_model_trainer import model_trainer
from mlProject.stage.stage_05_model_evaluation import model_evaluation


@pipeline
def training_pipeline():
    data_path = data_ingestion()
    data_frame = data_validation(data_path=data_path)
    train, test = data_transformation(data_frame = data_frame)
    model_trainer(train, test)
   


if __name__ == "__main__":
    training_pipeline()