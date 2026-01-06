# Credit Card Fraud Detection (PyTorch)

An end to end fraud detection project using PyTorch on anonymized credit card transaction data. The model is trained under extreme class imbalance and evaluated using production style ranking and review queue metrics.

---

## Dataset
- **V1–V28**: PCA transformed features used to preserve customer and transaction privacy  
- **Time**: Seconds elapsed between transactions  
- **Amount**: Transaction value  
- **Class**: Fraud label (1 = fraud, 0 = legitimate)  

Fraud transactions represent a very small fraction of the data.

---

## Approach
- Supervised learning with a PyTorch MLP for tabular data  
- GPU accelerated training  
- Weighted binary cross entropy loss to handle class imbalance  
- Time and Amount scaled using training only statistics to avoid data leakage  

The model outputs a fraud risk score for each transaction.

---

## Evaluation
Accuracy is not meaningful for fraud detection. This project focuses on:

- **Precision Recall AUC (PR AUC)**  
- **ROC AUC**  
- **Threshold tuning** for precision recall tradeoffs  
- **Top K review queue simulation**, reflecting real analyst workflows  

---

## Results
- **Test PR AUC**: ~0.73  
- **Recall**: ~80 percent at a practical operating threshold  
- Strong lift in top K transaction review compared to random selection  

---

## Real World Usage
In practice, the system would:
1. Score transactions by fraud risk  
2. Rank transactions from highest to lowest risk  
3. Surface the top K transactions for analyst review  

This balances fraud detection performance with operational constraints.

---

## Tech Stack
- Python  
- PyTorch (CNN) 
- NumPy  
- Pandas  
- Matplotlib  

---

## How to Run
1. Open the notebook in Kaggle with GPU enabled  
2. Run all cells sequentially  
3. Review evaluation outputs and plots  

---

## Author
Yash Soni
