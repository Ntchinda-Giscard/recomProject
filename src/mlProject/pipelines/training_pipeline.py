from zenml import pipeline
from mlProject.stage.stage_01_data_ingestion import data_loader
from mlProject.stage.stage_02_data_validation import data_validation
from mlProject.stage.stage_03_data_transformation import data_processing
# from mlProject.stage.stage_04_model_trainer import model_trainer
# from mlProject.stage.stage_05_model_evaluation import model_evaluation


@pipeline
def training_pipeline():
    data_path = data_loader()
    validation_status = data_validation(data_path=data_path)
    X_tarin, X_test, y_train, y_test = data_transformation(validation_status = validation_status)
    # model_trainer(train, test)
   


if __name__ == "__main__":
    training_pipeline()