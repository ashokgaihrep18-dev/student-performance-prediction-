# Student Exam Score Predictor

A beginner-friendly end-to-end machine learning project: predict a student's exam score from study habits, attendance, and background factors, then serve the model in a small web app.

**Live demo:** _ https://axcer26pbifp3vkr4ihas8.streamlit.app/

<img width="650" height="863" alt="image" src="https://github.com/user-attachments/assets/83b007c7-0844-4376-979d-1a4f3c1e3cc1" />


## What it does

Enter a student's details (hours studied, attendance, sleep, parental involvement, and more) and the app predicts the exam score using a trained Linear Regression model.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit

## Dataset

[Student Performance Factors](https://www.kaggle.com/) (Kaggle, CC0 public domain): 6,607 students, 19 input features, target `Exam_Score`. Search for "Student Performance Factors" on Kaggle.
The data appears to be synthetic, so this project demonstrates the machine learning workflow rather than real-world conclusions.

## Workflow

1. Loaded the CSV with Pandas and inspected shape, types, and statistics.
2. Explored missing values, duplicates, score distribution, category counts, and correlations.
3. Cleaned the data: filled missing categorical values (`Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`) with the mode, and encoded text columns with one-hot encoding (`get_dummies`).
4. Split the data 80/20 into training and test sets.
5. Trained a Linear Regression model.
6. Evaluated with MAE, MSE and R².
7. Saved the model and its column order with Joblib.
8. Built a Streamlit app that applies the same encoding and predicts new inputs.

## Results

| Version | MAE | MSE | R² |
|---|---|---|---|
| All data (6,607 rows) | 0.45 | 3.26 | 0.77 |
| Without extreme scores above 80 (6,564 rows) | 0.31 | 0.72 | 0.935 |

**Why two rows?** A few students (43 rows, about 0.65%) had very high scores (above 80) that the available features cannot explain. A handful of these inflated MSE and lowered R². I removed them for the final model and report both results openly. The final model is meant for typical scores (up to 80), not extreme ones.

## Key findings

- Attendance and hours studied are the strongest numeric drivers of exam score.
- Low parental involvement and low access to resources lower the predicted score the most (about 2 marks each, compared with High).
- These are associations in this dataset, not proof of cause.



## Project files

| File | Purpose |
|---|---|
| `student_score_model.ipynb` | Data exploration, cleaning, training, evaluation |
| `app.py` | Streamlit web app |
| `model.pkl` | Trained Linear Regression model |
| `columns.pkl` | Column names and order the model expects |
| `StudentPerformanceFactors.csv` | Dataset |
| `requirements.txt` | Python dependencies |

## Limitations and next steps

- Linear Regression assumes linear relationships; try Random Forest or Gradient Boosting and compare.
- The dataset is likely synthetic, so test on real data before drawing real conclusions.
- Add input validation and feature importance charts to the app.

## Author

Ashok, Nepal. Learning machine learning and data engineering.
