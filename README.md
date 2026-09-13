\# Online Fraud Detection Using Machine Learning



\## Project Overview



This project uses Machine Learning to detect fraudulent online transactions.



A Random Forest model is used to classify transactions as either:



\- Legitimate

\- Fraudulent



The project also includes a Streamlit web application that allows users to enter transaction details and get a fraud prediction.



\## Machine Learning Model



The final model is a Random Forest classifier with 200 trees and balanced class weights.



The model gives a fraud probability for each transaction.



\- Below 50% → LEGITIMATE

\- 50% or above → FRAUD



\## Model Performance



| Metric | Score |

|---|---:|

| Accuracy | 99.95% |

| Fraud Precision | 94.44% |

| Fraud Recall | 71.58% |

| Fraud F1-Score | 81.44% |

| ROC-AUC | 94.30% |

| PR-AUC | 81.89% |



\## Streamlit Application



The Streamlit application provides:



\- Fraud probability

\- Fraud/Legitimate prediction

\- Risk level

\- Feature importance



\### Risk Levels



\- Below 50% → LOW / LEGITIMATE

\- 50%–79.99% → MEDIUM / FRAUD

\- 80% or above → HIGH / FRAUD



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Imbalanced-learn

\- Joblib

\- Matplotlib

\- Seaborn

\- Streamlit



\## Project Structure



```text

online-fraud-detection-ml/

├── app/

│   └── app.py

├── models/

│   ├── final\_random\_forest.pkl

│   └── fraud\_threshold.pkl

├── notebooks/

├── requirements.txt

├── .gitignore

└── README.md

