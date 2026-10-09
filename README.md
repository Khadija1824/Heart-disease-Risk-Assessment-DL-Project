# 🫀 AI-Powered Heart Disease Prediction Dashboard

An end-to-end deep learning web application that predicts cardiovascular disease risk using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras and deployed via an interactive **Streamlit** dashboard.

---

## 🚀 Features
* **Real Kaggle Data Integration:** Trained on the standard UCI Heart Disease dataset.
* **Deep Learning Classification:** Utilizes a fully connected Keras sequential neural network with dropout regularization to prevent overfitting.
* **Interactive UI:** A user-friendly Streamlit sidebar enabling doctors or patients to adjust clinical vitals and evaluate risk instantly.
* **Probability Metrics:** Outputs live confidence percentage scores alongside diagnostic assessments.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing & Scaling:** Pandas, Scikit-learn (`StandardScaler`), Joblib
* **Web Interface:** Streamlit

📊 How to Use
Once the Streamlit app launches in your browser, use the sidebar sliders and dropdowns to input patient clinical parameters (such as Age, Resting Blood Pressure, Cholesterol, and Max Heart Rate).

Click the "Evaluate Heart Risk" button.

View the predicted disease probability percentage and the corresponding clinical risk assessment.

## 📈 Model Performance & Architecture
* **Architecture:** 3-layer Feedforward Neural Network (Dense 16 (ReLU) $\rightarrow$ Dropout (0.2) $\rightarrow$ Dense 8 (ReLU) $\rightarrow$ Dense 1 (Sigmoid))
* **Optimizer:** Adam Optimizer
* **Loss Function:** Binary Crossentropy
* **Test Accuracy:** ~85.2%

## 🔮 Future Enhancements
*  Integrate a Confusion Matrix & ROC-AUC curve visualization tab in Streamlit.
*  Compare Keras ANN performance against an XGBoost classifier.
*  Deploy live via Streamlit Community Cloud for public web access.
