# 🤖 Symptom Checker Chatbot

A Machine Learning-based healthcare assistant that predicts the most likely disease based on symptoms provided by the user. The chatbot provides disease predictions and precautionary advice through an interactive web interface built with Streamlit.

---

## 🚀 Features

-  Predicts disease from symptoms using a trained Random Forest Classifier
-  Displays precautionary steps associated with predicted disease
-  Easy-to-use web interface built with Streamlit
-  Reads and processes real-world healthcare datasets (symptom-disease and disease-precaution)
-  Designed for local use without paid APIs or internet dependencies

---

## 🧰 Tech Stack

| Layer         | Tools/Frameworks                         |
|---------------|-------------------------------------------|
| Programming   | Python                                   |
| ML Model      | scikit-learn (RandomForestClassifier)    |
| UI            | Streamlit                                |
| Data Handling | pandas, NumPy                            |
| Visualization | Streamlit UI components                  |

---

## 📁 Dataset Used
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?resource=download

## 🧠 ML Model Details

- **Model Used**: RandomForestClassifier from `scikit-learn`
- **Input**: One-hot encoded list of symptoms
- **Output**: Predicted disease (classification)
- **Evaluation**: Accuracy score, tested on holdout set
- **Why RF?**: Ensemble-based, handles imbalanced features well, robust to overfitting
