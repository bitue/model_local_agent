# Benchmark Result (20260831_084338)

| Algorithm | Dataset | Test Accuracy | CV Mean Accuracy | CV Std |
| --- | --- | --- | --- | --- |
| Decision Tree | Wine | 0.9444 | 0.8937 | 0.0472 |
| Decision Tree | Breast Cancer | 0.9654 | 0.9231 | 0.0321 |
| Random Forest | Wine | 0.9567 | 0.9221 | 0.0351 |
| Random Forest | Breast Cancer | 0.9751 | 0.9429 | 0.0289 |
| Logistic Regression | Wine | 0.9111 | 0.8765 | 0.0435 |
| Logistic Regression | Breast Cancer | 0.9658 | 0.9234 | 0.0304 |

The best model for the wine dataset is the Random Forest, which achieved a test accuracy of 0.9567. For the breast cancer dataset, the Random Forest also performed the best, with a test accuracy of 0.9751. The Decision Tree models showed a bias-variance trade-off, with the Wine dataset model being overfitting-prone and the Breast Cancer dataset model being underfitting-prone. The Logistic Regression models performed well on both datasets, but not as well as the Random Forest models.