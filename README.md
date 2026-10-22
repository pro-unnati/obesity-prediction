# Obesity Prediction System

A Machine Learning-based web application that predicts a person's weight category based on their health and lifestyle information. The application uses a trained Machine Learning model and Streamlit to provide an interactive prediction experience.

## Features

- Predicts weight categories using user-provided information.
- Classifies individuals into weight categories such as Underweight, Normal Weight, Overweight, and Obesity.
- Interactive web interface built using Streamlit.
- Uses a pre-trained Machine Learning model for predictions.
- Includes a separate Python script for model training.

## Weight Categories

The model predicts a weight category based on the classes it was trained to recognize, including:

- Underweight
- Normal Weight
- Overweight
- Obesity

*The exact categories depend on the target labels used during model training.*

## Technologies Used

- **Python** – Programming language
- **Pandas** – Data handling and preprocessing
- **NumPy** – Numerical operations, if used
- **Scikit-learn** – Machine Learning
- **Streamlit** – Web application interface
- **Joblib / Pickle** – Loading the trained model, depending on the implementation

## Project Structure

```text
Obesity_Prediction/
│
├── app.py
├── train_model.py
├── obesity_data.csv
├── obesity_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

1. The user enters the required health and lifestyle information in the application.
2. The input data is processed according to the format expected by the trained model.
3. The Machine Learning model predicts the corresponding weight category.
4. The application displays the predicted category to the user.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Obesity_Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Obesity_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser. Enter the required information to view the predicted weight category.

## Model Training

To train the model again using the available dataset, run:

```bash
python train_model.py
```

This script is intended to train the Machine Learning model using `obesity_data.csv` and save the trained model as `obesity_model.pkl`, depending on the implementation in the training script.

## Dataset

The project includes `obesity_data.csv`, which is used for model training. The dataset contains the information required by the model to learn patterns associated with weight categories.

## Disclaimer

This project is developed for educational purposes. Its predictions are based on the trained model and should not be considered a medical diagnosis or a substitute for professional medical advice.

## Author

**Unnati Gupta**

B.Tech Information Technology Student
