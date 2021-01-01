Here’s the updated README with a **Project Structure** section added under the **Tech Stack** for clarity:

---

# Movie Recommendation System with MLOps Integration

**Overview**  
An end-to-end machine learning system for personalized movie recommendations, powered by a **content-based algorithm** built with TensorFlow/Keras. The project leverages modern MLOps tools like ZenML, MLflow, Evidently, and BentoML for robust pipeline orchestration, validation, and deployment.

---

## 🔥 **What’s Blazing Under the Hood?**

- **TensorFlow/Keras Content-Based Algorithm**: Neural networks that analyze user preferences and movie metadata to predict ratings with surgical precision.
- **ZenML Orchestration**: Unifies the ML workflow with automated pipelines, artifact tracking, and tool integrations.
- **Evidently**: Validates data quality and model performance in production-like scenarios.
- **MLflow**: Tracks experiments, logs metrics, and manages model registry.
- **BentoML**: Streamlines cloud deployment with containerized model serving.

---

## 🏗️ **Project Architecture**

### Pipeline Definition

1. **Data Ingestion**:

   - Collects movie ratings and metadata from CSV datasets.

2. **Data Validation**:

   - **Schema Validation**: Ensures CSV files match predefined schemas (column names, data types).
   - **Data Range Check**: Validates ratings fall within 0.5–5.0 to prevent target leakage.

3. **Data Processing**:

   - **User Feature extraction**: Extract user feature based preferences and past ratings.
   - **Movie Feature extraction**: Extract features from movie based on Tags, Genre and past Ratings.

4. **Model Training**:

   - Dual-tower neural network architecture:
     - **User Tower**: Embeds user preferences.
     - **Movie Tower**: Encodes movie attributes.
   - <img src="./assets/model.png" alt="Model Architecture" style="width:500px;height:280px;">
   - Logs metrics/artifacts to MLflow and registers models in the MLflow registry via ZenML.

5. **Model Evaluation**:

   - Computes accuracy, RMSE, and user-specific ranking metrics.

6. **Model Registry**:

   - Version-controlled model storage in MLflow.

7. **BentoML Promotion**:
   - Packages validated models for cloud deployment.

---

## 🛠️ **Tech Stack**

- **ML Framework**: TensorFlow 2.x, Keras
- **MLOps**: ZenML, MLflow (experiment tracking/model registry), Evidently (validation), BentoML (deployment)
- **Data Tools**: Pandas, NumPy, Scikit-learn

---

## 📂 Project Structure

```bash
END-END-MLFLOW/
├── artfacts/
├── logs/
├── config/
│   ├──config.yaml      # Project configuration for each pipeline step
├── logs/
├── mlruns/
├── research/
├── src/
│   ├──mlProject/
│       ├── steps/      # ZenML steps pipeline definition
│       │   ├── step_01_data_loader.py
│       │   ├── step_02_data_validation.py
│       │   ├── step_03_data_processing.py
│       │   ├── step_04_model_trainer.py
│       │   ├── step_05_model_evaluation.py
│       │   └── step_06_model_promotion.py
│       │
│       ├── pipelines/   # ZenML pipeline definition
│       │   ├── training_pipeline.py
│       │   └── ...
│       ├── components/
│       │   ├── data_loader.py
│       │   ├── data_validation.py
│       │   ├── data_processing.py
│       │   ├── model_trainer.py
│       │   ├── model_evaluation.py
│       │   ├── model_service.py
│       │   └── model_promotion.py
│       └── ...
├── assets/
│   ├── model.png      # Model architecture diagram
│   └── pipeline.png   # Pipeline workflow visualization
├── config/
│   ├── configuration.py     # Data schema definitions
│   └── ...            # Environment variables/paths
├── tests/             # Unit/integration tests
├── requirements.txt   # Project dependencies
├── bentofile.yaml     # Bentoml yaml bento builder configuration
├── schema.yaml        # Dataset schema definition
└── main.py            # Pipeline execution script
```

---

## 🚀 **Running This Project**

### Prerequisites

- Python 3.10
- ZenML Cloud Account
- BentoML Cloud Account
- MLflow Server (e.g., hosted on DagsHub)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ntchinda-Giscard/recomProject.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Connect to ZenML Server:
   ```bash
   zenml login "YOUR_SERVER_URL"
   ```

### ZenML Stack Setup

```bash
# Register components
zenml model-deployer register bentoml_deployer --flavor=bentoml
zenml model-registry register mlflow_registry --flavor=mlflow
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml data-validator register evidently_validator --flavor=evidently

# Create and activate stack
zenml stack register my_stack \
    -d bentoml_deployer \
    -r mlflow_registry \
    -e mlflow_tracker \
    -v evidently_validator \
    -o default \
    -a default

zenml stack set my_stack
```

### Execution

1. Run the pipeline:
   ```bash
   python main.py
   ```
   Pipeline overview:
   - <img src="./assets/pipeline.png" alt="Pipeline Workflow" style="width:500px;height:280px;">
2. Deploy to BentoML Cloud:

   ```bash
   bentoml cloud login \
       --api-token 'YOUR_API_TOKEN' \
       --endpoint 'YOUR_SERVER_URL'

   bentoml deploy --name recommend-system .
   ```

---

## 📄 **License**

MIT License. See `LICENSE` for details.

**Need Help?** Open an issue or contact [@Ntchinda-Giscard](https://www.linkedin.com/in/ntchinda-giscard-62abb71bb).
