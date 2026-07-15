# 🎓 Student Exam Performance Indicator

> An end-to-end Machine Learning project that predicts a student's **Math Score** based on demographic and academic input factors — deployed on both **AWS Elastic Beanstalk** and **Azure App Service** with a full **CI/CD pipeline** via GitHub Actions.

---

## 📌 Problem Statement

Given a student's background information (gender, ethnicity, parental education level, lunch type, test preparation course, reading score, and writing score), predict their **Math Score** (0–100).

This is a **supervised regression** problem solved using multiple ML algorithms with automated hyperparameter tuning and the best model is selected automatically.

---

## 🚀 Live Demo

| Platform | Link |
|----------|------|
| Azure App Service | Deployed via GitHub Actions CI/CD |
| AWS Elastic Beanstalk | Deployed via `.ebextensions` config |

---

## 🏗️ Project Architecture

```
User → Web Form (Flask) → Predict Pipeline → Preprocessor + Best ML Model → Math Score
```

The training pipeline:
```
Raw CSV → Data Ingestion → Data Transformation → Model Training (7 models + GridSearchCV) → Best Model Saved → Artifacts
```

---

## 📁 Folder Structure

```
├── .ebextensions/
│   └── python.config              # AWS Elastic Beanstalk WSGI config
├── .github/
│   └── workflows/
│       └── main_studentssperformance3.yml  # Azure CI/CD GitHub Actions workflow
├── artifacts/
│   ├── data.csv                   # Raw dataset copy
│   ├── train.csv                  # Training split (80%)
│   ├── test.csv                   # Test split (20%)
│   ├── model.pkl                  # Best trained model (serialized)
│   └── preprocessor.pkl           # Fitted preprocessing pipeline (serialized)
├── notebook/
│   ├── data/
│   │   └── stud.csv               # Source dataset (~1000 student records)
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # Exploratory Data Analysis
│   └── 2. MODEL TRAINING.ipynb    # Model experimentation & comparison
├── src/
│   ├── __init__.py
│   ├── exception.py               # Custom exception class with detailed error tracing
│   ├── logger.py                  # Timestamped file-based logging
│   ├── utils.py                   # Utility functions: save/load objects, model evaluation
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Reads raw data, performs train/test split
│   │   ├── data_transformation.py # Builds sklearn preprocessing pipeline & transforms data
│   │   └── model_trainer.py       # Trains 7 regression models, tunes hyperparams, saves best
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py      # Orchestrates end-to-end training
│       └── predict_pipeline.py    # Loads artifacts, accepts new input, returns prediction
├── templates/
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form & results page
├── app.py                         # Flask application entry point
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup (installable as `mlproject`)
└── README.md
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
- Reads the raw student CSV dataset
- Performs an 80/20 random train-test split (`random_state=42`)
- Saves `train.csv`, `test.csv`, and `data.csv` under `artifacts/`

### 2. Data Transformation (`data_transformation.py`)
Builds a **scikit-learn `ColumnTransformer`** with two sub-pipelines:

| Feature Type | Columns | Transformations |
|---|---|---|
| **Numerical** | `reading_score`, `writing_score` | `SimpleImputer(median)` → `StandardScaler` |
| **Categorical** | `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` | `SimpleImputer(most_frequent)` → `OneHotEncoder` → `StandardScaler` |

The fitted preprocessor object is serialized to `artifacts/preprocessor.pkl`.

### 3. Model Training (`model_trainer.py`)
Seven regression models are trained and evaluated with **GridSearchCV (3-fold CV)**:

| Model | Key Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Decision Tree Regressor | `criterion` |
| Random Forest Regressor | `n_estimators` |
| Gradient Boosting Regressor | `n_estimators`, `learning_rate`, `subsample` |
| XGBoost Regressor | `n_estimators`, `learning_rate` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |
| AdaBoost Regressor | `n_estimators`, `learning_rate` |

- The model with the **highest R² score** on the test set is saved to `artifacts/model.pkl`
- A minimum R² threshold of **0.6** is enforced; otherwise an exception is raised

### 4. Prediction Pipeline (`predict_pipeline.py`)
- Loads `model.pkl` and `preprocessor.pkl` from `artifacts/`
- Accepts a `CustomData` object (one student's inputs) as a Pandas DataFrame
- Returns the predicted math score

---

## 🌐 Web Application

Built with **Flask**, the app exposes two routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing / home page |
| `/predictdata` | GET | Renders the prediction form |
| `/predictdata` | POST | Accepts form input, runs prediction, displays result |

**Input fields on the form:**
- Gender (Male / Female)
- Race / Ethnicity (Group A – E)
- Parental Level of Education (6 categories)
- Lunch Type (Standard / Free-Reduced)
- Test Preparation Course (None / Completed)
- Reading Score (0–100)
- Writing Score (0–100)

**Output:** Predicted Math Score

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.7+ |
| Web Framework | Flask |
| ML / Data | scikit-learn, CatBoost, XGBoost, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn (notebooks) |
| Serialization | Pickle |
| Logging | Python `logging` module (timestamped log files) |
| Cloud (AWS) | AWS Elastic Beanstalk |
| Cloud (Azure) | Azure App Service |
| CI/CD | GitHub Actions |
| Packaging | setuptools (`setup.py`) |

---

## ⚙️ Getting Started Locally

### Prerequisites
- Python 3.7+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aaadityasngh/End-To-End-Machine-Learning-Project-with-AWS-Azure-Deployment.git
cd End-To-End-Machine-Learning-Project-with-AWS-Azure-Deployment

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies (also installs the src package in editable mode)
pip install -r requirements.txt
```

### Training the Model

```bash
python src/components/data_ingestion.py
```

This will:
1. Read `notebook/data/stud.csv`
2. Split into train/test and save under `artifacts/`
3. Build and apply the preprocessing pipeline, saving `artifacts/preprocessor.pkl`
4. Train all 7 models with hyperparameter tuning
5. Save the best model to `artifacts/model.pkl`
6. Print the best model's R² score

### Running the Web App

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser.

---

## ☁️ Deployment

### AWS Elastic Beanstalk

The `.ebextensions/python.config` sets the WSGI entry point:

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

Deploy using the AWS EB CLI:
```bash
eb init -p python-3.7 mlproject
eb create mlproject-env
eb deploy
```

### Azure App Service (CI/CD via GitHub Actions)

The workflow `.github/workflows/main_studentssperformance3.yml` automatically:
1. **Builds** the app (sets up Python 3.7, installs requirements)
2. **Packages** and uploads the artifact
3. **Deploys** to Azure App Service (`studentssperformance3`) on every push to `main`

Set the `AZUREAPPSERVICE_PUBLISHPROFILE` secret in your GitHub repository settings to enable automatic deployments.

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender |
| `race_ethnicity` | Categorical | Racial/ethnic group (A–E) |
| `parental_level_of_education` | Categorical | Highest education level of parents |
| `lunch` | Categorical | Standard or free/reduced lunch |
| `test_preparation_course` | Categorical | Whether test prep was completed |
| `reading_score` | Numerical (0–100) | Score in reading exam |
| `writing_score` | Numerical (0–100) | Score in writing exam |
| `math_score` | Numerical (0–100) | **Target variable** |

Source: Commonly used [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset (~1000 records).

---

## 📈 Key Engineering Decisions

- **Modular design**: Each ML stage (ingestion, transformation, training) is a standalone, reusable Python class — making it easy to swap components independently.
- **Custom exception handling**: Detailed error messages include the file name and line number, making debugging straightforward.
- **Automated model selection**: All 7 models are evaluated automatically; the best one is saved without manual intervention.
- **Serialized artifacts**: Both the preprocessing pipeline and the model are persisted as `.pkl` files, ensuring the prediction pipeline uses exactly the same transformations as training.
- **Dual-cloud deployment**: The app is deployable on both AWS and Azure, demonstrating cloud-agnostic design.

---

## 🔧 Project Setup Details

The project is packaged via `setup.py`, making the `src/` directory importable as the `End-To-End-ML-pipeline` package across the entire codebase (the `-e .` in `requirements.txt` achieves this).

---

## 👨‍💻 Author

**Aditya Singh**
- Email: sumitsharma9098660389@gmail.com
- GitHub: [@sumitsharma12321](https://github.com/Sumitsharma12321)

---

## 📄 License

This project is open-source and available for learning and educational purposes.
