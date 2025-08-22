# knn-diabetes-prediction
A machine learning project that predicts diabetes using the Pima Indians Diabetes dataset with the K-Nearest Neighbors (KNN) algorithm. Includes preprocessing (missing values, scaling), hyperparameter tuning, evaluation metrics (accuracy, confusion matrix, ROC, AUC), and visualizations.

---

# KNN Diabetes Prediction Project

This project uses the **Pima Indians Diabetes Dataset** to predict whether a patient has diabetes or not using the **K-Nearest Neighbors (KNN)** algorithm.

## 📂 Project Files
- `KNN_model.ipynb` : Jupyter Notebook containing the full implementation of KNN on the dataset.
- `diabetes.csv` : Dataset file (Pima Indians Diabetes dataset from Kaggle).

## ⚙️ Requirements
Make sure you have Python 3.8+ and install the following libraries:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## 📊 Steps in the Notebook
1. **Data Loading**
   - Load the diabetes dataset (CSV).
2. **Data Preprocessing**
   - Replace invalid `0` values with NaN in columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI.
   - Impute missing values (mean/median/KNNImputer).
   - Apply **StandardScaler** for feature scaling.
3. **Train/Test Split**
   - Split data into training and testing sets.
4. **Model Training (KNN)**
   - Train KNN classifier on the scaled dataset.
   - Tune the number of neighbors (`K`) using cross-validation (GridSearchCV).
5. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - ROC Curve & AUC Score
   - Accuracy vs K plot (Elbow method)
6. **Visualization**
   - Confusion Matrix plot
   - ROC Curve
   - Accuracy vs K graph

## 📈 Example Outputs
- Model Accuracy: ~74–75% (varies with parameters)
- Best K value: Found using GridSearchCV
- Confusion Matrix, ROC curve, and Accuracy vs K plots

## 🚀 How to Run
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook KNN_model.ipynb
   ```
2. Run all cells step by step.
3. View evaluation metrics and plots at the end.

## 🔮 Improvements
- Try different distance metrics (`euclidean`, `manhattan`).
- Experiment with feature selection (remove noisy columns).
- Handle class imbalance using SMOTE or class weights.
- Compare with other ML models (RandomForest, SVM, Logistic Regression).

---

📌 Author: *Waleed Afridi*  
📅 Date: August 2025




