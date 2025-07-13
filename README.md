# UFC Fight Predictor

## Overview
The UFC Fight Predictor is a Python-based web application that allows users to select two UFC fighters and receive a prediction for the fight's outcome, including the predicted winner, method of victory, and round. This project serves as a demonstration of a machine learning pipeline, from data collection and preprocessing to model training and deployment as a Flask web application.

## Features
- **Fighter Selection**: Users can easily select two different UFC fighters from a dropdown list for a hypothetical matchup.
- **Fight Prediction**: Predicts the winner of the selected fight, along with the likely method of victory (e.g., Knockout, Submission, Decision) and the round in which it will occur.
- **Data Pipeline**: Includes scripts for:
  - Data Collection: Gathers initial fighter and fight data (currently uses sample data).
  - Data Preprocessing: Cleans and transforms raw data for model consumption.
  - Feature Engineering: Creates relevant features for machine learning models from the preprocessed data, including fighter statistics and comparative attributes.
- **Machine Learning Models**: Utilizes trained models to make predictions based on fighter attributes and historical performance.
- **Web Interface**: A simple and intuitive Flask web application provides the user interface for interactions.

## Technologies Used
- **Python**: The core programming language.
- **Flask**: Web framework for building the application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning model training and prediction (e.g., classification models, data scaling).
- **Joblib**: For saving and loading trained machine learning models and scalers.
- **HTML/CSS/JavaScript**: For the frontend web interface.

## Local Setup and Installation

To get this project running on your local machine, follow these steps:

### 1. Clone the Repository:
```bash
git clone <repository-url>
cd ufc-fight-predictor
```

### 2. Create a Virtual Environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Collect Raw Data:
This script generates initial fighter and fight data (currently sample data).
```bash
python data_collection.py
```
This will create `data/fighters.csv` and `data/fights.csv`.

### 5. Preprocess Data:
This script cleans and prepares the raw data.
```bash
python data_preprocessing.py
```
This will create `data/processed_fighters.csv` and `data/processed_fights.csv`.

### 6. Perform Feature Engineering:
This script creates advanced features for the models and generates `enhanced_fighters.csv` and `fight_pairs.csv`.
```bash
python feature_engineering.py
```

### 7. Train Models:
This script trains the machine learning models and saves them (.pkl files) in the `models/` directory.
```bash
python model_training.py
```

### 8. Run the Flask Application:
Start the web server.
```bash
python app.py
```
The application will typically run on `http://127.0.0.1:5000/`. Open this URL in your web browser.

## Usage

Once the Flask application is running:

1. Navigate to `http://127.0.0.1:5000/` in your web browser.
2. Use the dropdown menus to select two different fighters.
3. Click the "Predict Fight" button to see the predicted outcome.

## Project Structure
```
ufc-fight-predictor/
├── app.py                 # Flask web application
├── data_collection.py     # Data collection script
├── data_preprocessing.py  # Data preprocessing script
├── feature_engineering.py # Feature engineering script
├── model_training.py      # Model training script
├── requirements.txt       # Python dependencies
├── data/                  # Data directory
│   ├── fighters.csv
│   ├── fights.csv
│   ├── processed_fighters.csv
│   ├── processed_fights.csv
│   ├── enhanced_fighters.csv
│   └── fight_pairs.csv
├── models/                # Trained models directory
│   ├── winner_model.pkl
│   ├── method_model.pkl
│   ├── round_model.pkl
│   └── scaler.pkl
├── templates/             # HTML templates
└── static/               # CSS and JavaScript files
```
