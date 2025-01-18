from zenml import pipeline
from mlProject.steps.step_01_data_loader import data_loader
from mlProject.steps.step_02_data_validation import data_validation
from mlProject.steps.step_03_data_processing import data_processing
# from mlProject.stage.stage_04_model_trainer import model_trainer
# from mlProject.stage.stage_05_model_evaluation import model_evaluation


@pipeline
def training_pipeline():
    data_path = data_loader()
    validation_status = data_validation(data_path=data_path)
    X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating = data_processing(validation_status = validation_status)
    # model_trainer(train, test)
   


if __name__ == "__main__":
    training_pipeline()