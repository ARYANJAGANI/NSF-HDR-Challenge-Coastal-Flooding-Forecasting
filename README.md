# Coastal Flooding Forecasting using Machine Learning & LSTM  
### NSF HDR Challenge | IS 755 – Intelligent Systems

---

## 📌 Project Overview

Coastal flooding poses a significant environmental and societal risk due to rising sea levels, storm surges, and climate variability. Accurate short-term flood forecasting is essential for early warning systems, infrastructure planning, and disaster mitigation.

This project was developed as part of the **NSF HDR Coastal Flooding Forecasting Challenge** and **IS 755 (Intelligent Systems)** at the **University of Maryland, Baltimore County (UMBC)**. The goal is to predict coastal flooding events using long-term tide gauge observations and evaluate model performance on the official **Codabench leaderboard**.

We implement and compare:
- A **baseline XGBoost model**
- A **Long Short-Term Memory (LSTM)** deep learning model to capture temporal dependencies in sea-level data

---

## 👤 Author

**Aryan Jagani**  
Master’s Student – Information Systems  
University of Maryland, Baltimore County (UMBC)

---

## 🎯 Objectives

- Perform exploratory data analysis on long-term coastal sea-level data  
- Engineer meaningful temporal features for flood prediction  
- Build and evaluate machine learning and deep learning models  
- Forecast flood occurrence for a **14-day horizon** using a **7-day historical window**  
- Submit predictions to the **Codabench leaderboard**  

---

## 🌊 Dataset Description

- **Source:** Long-term tide gauge observations (1950–2020)  
- **Stations Used (subset):**
  - Annapolis  
  - Atlantic City  
  - Charleston  
  - Washington  
  - Wilmington  

### Data Characteristics
- Hourly sea-level measurements (meters)
- Converted from MATLAB datenum to Python datetime
- Aggregated to **daily resolution**
- Includes station metadata (latitude, longitude)

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings from EDA:
- Strong **seasonal and tidal variability**
- Long-term upward trends consistent with sea-level rise
- Presence of extreme values corresponding to storm surge events
- Moderate to high correlation between geographically close stations

EDA confirmed the dataset’s suitability for **sequence-based models** such as LSTM.

---

## 🧪 Feature Engineering

To capture short-term and medium-term dynamics, the following features were engineered:

- Daily mean sea level  
- 3-day rolling mean  
- 7-day rolling mean  

### Flood Label Definition
A flood event is labeled as:
- `1` if the daily maximum sea level exceeds a station-specific threshold  
- `0` otherwise  

---

## 🧠 Problem Formulation

- **Task:** Multi-output binary classification  
- **Input:** 7-day historical window  
- **Output:** Flood occurrence for the next 14 consecutive days  

Each forecast day is modeled independently.

---

## 🤖 Models Implemented

### 1️⃣ Baseline Model – XGBoost
- 14 independent XGBoost classifiers (one per forecast day)
- Strong baseline for structured tabular data
- Key parameters:
  - Number of estimators: 100
  - Max depth: 5
  - Learning rate: 0.1

### 2️⃣ Proposed Model – LSTM
- Designed to capture temporal dependencies in sea-level data
- Architecture:
  - Input shape: (7 timesteps × 3 features)
  - LSTM layer with 50 hidden units
  - Dropout (0.2)
  - Dense layer with sigmoid activation
- Training:
  - Binary cross-entropy loss
  - Adam optimizer
  - 10 epochs
  - Batch size: 32
- Separate LSTM model trained for each forecast horizon (14 total)

---

## 📊 Model Evaluation

### Metrics Used
- Accuracy  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrix  
- ROC–AUC  

### Key Results
- LSTM achieved:
  - **ROC–AUC: 0.871**
  - Competitive F1-score compared to baseline
- XGBoost slightly outperformed LSTM on some metrics
- LSTM demonstrated strong ability to model temporal dependencies

---

## 🏆 Codabench Submission

- Successfully submitted predictions to the **Codabench leaderboard**
- Achieved:
  - **F1 Score: 0.94**
  - **Accuracy: 0.89**
- Performance ranked competitively among submitted models

---

## ⚠️ Limitations

- Class imbalance affects MCC and minority flood events
- Higher computational cost for LSTM models
- Independent models for each forecast day increase complexity
- Limited use of meteorological variables

---

## 🚀 Future Work

- Incorporate weather and climate variables
- Use a unified multi-output LSTM architecture
- Apply attention-based models or Transformers
- Improve class balancing strategies
- Extend forecasting horizon

---

## 🛠 Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  
- Matplotlib, Seaborn  
- Google Colab (GPU)  

---

## 📂 Repository Structure

