from zenml import pipeline, step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from mlProject.steps.step_01_data_loader import data_loader
from mlProject.steps.step_02_data_validation import data_validation
from mlProject.steps.step_03_data_processing import data_processing
from mlProject.steps.step_04_model_trainer import model_trainer
from mlProject.steps.step_05_model_evaluation import model_evaluation
from mlProject.steps.step_06_model_promotion import promote_to_bentoml

@pipeline
def training_pipeline():
    data_path = data_loader()
    validation_status = data_validation(data_path=data_path)
    X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating = data_processing(validation_status = validation_status)
    model, history = model_trainer(X_train_user, X_train_movie, y_train_rating)
    mlflow_register_model_step(
        model=model,
        name=f"RecommendNet"
    )
    model_evaluation(
        X_val_user, X_val_movie, y_val_rating,
        model)
    promote_to_bentoml()
    
   


if __name__ == "__main__":
    training_pipeline()