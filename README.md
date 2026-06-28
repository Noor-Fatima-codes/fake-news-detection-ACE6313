# Fake News Detection | ACE6313 | SDG 16

## Group Members
- Noor Fatima
- Yasmeen Abdelkader

## Dataset
ISOT Fake News Dataset
Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## How to Run
1. Install requirements: pip install pandas scikit-learn nltk seaborn matplotlib joblib wordcloud
2. Download dataset and place fake.csv and true.csv in the same folder
3. Run: python your_code.py

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (Best Model — F1: 0.9881)
- Gradient Boosting

## Results
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.9765 | 0.9776 |
| Decision Tree | 0.9187 | 0.9236 |
| Random Forest | 0.9626 | 0.9643 |
| SVM | 0.9875 | 0.9881 |
| Gradient Boosting | 0.9493 | 0.9514 |

## SDG Alignment
This project supports SDG 16 — Peace, Justice and Strong Institutions
by using ML to combat misinformation and fake news.
