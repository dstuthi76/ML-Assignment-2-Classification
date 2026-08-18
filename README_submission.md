# Machine Learning Assignment – 2
## Breast Cancer Wisconsin (Diagnostic) Classification

### a. Problem Statement

The objective is to implement multiple machine-learning classification models
on one public classification dataset, evaluate the models using Accuracy, AUC,
Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and
demonstrate the trained models through an interactive Streamlit application.

The application accepts a CSV test dataset, allows the evaluator to select a
classification model, displays the six required evaluation metrics, shows a
confusion matrix/classification report, and displays the prediction results.

### b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) (WDBC)

**Source:** UCI Machine Learning Repository, Dataset ID 17.

The dataset contains **569 instances and 30 continuous predictive features**.
The target is a binary diagnosis: malignant (M) or benign (B). The features
are computed from digitized images of fine needle aspirates of breast masses
and describe characteristics of cell nuclei.

The 30 feature groups include radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry and fractal dimension,
measured as mean, standard error and worst values.

The patient ID is not used as a predictive feature. Malignant is encoded as 1
and benign as 0 for evaluation.

Dataset citation:
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
DOI: 10.24432/C5DW2B.

### Experimental Setup

- Train/test split: 80% / 20%
- Random state: 42
- Stratified split to preserve the class distribution
- StandardScaler is used inside pipelines for Logistic Regression and kNN.
- Decision Tree, Gaussian Naive Bayes and Random Forest use the original
  numerical features.
- kNN uses k = 5.
- Random Forest uses 200 trees.
- Positive class: malignant.

### c. GitHub Repository Link

**GitHub Repository:** [PASTE YOUR GITHUB REPOSITORY LINK HERE]

### d. Models Used

The assignment document names five required models even though one sentence
refers to “all 6 ML models”. The five explicitly listed models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9649 | 0.9942 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

### Observations on Model Performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Excellent overall performance with 96.49% accuracy and 0.9960 AUC. It gives a strong balance between precision and recall and is highly suitable for this dataset. |
| Decision Tree | The lowest overall performance among the five models in this experiment. Its 92.98% accuracy and 0.9246 AUC indicate that a single tree has higher variance and is less stable than the ensemble approach. |
| kNN | Strong performance with 95.61% accuracy and 0.9823 AUC. Standardization is important because kNN is distance-based. It achieves good recall while maintaining high precision. |
| Naive Bayes | Good AUC of 0.9934 and 93.86% accuracy. Its precision for the malignant class is 1.0000 in this test split, but recall is lower at 0.8333, so it misses some malignant cases. |
| Random Forest (Ensemble) | Excellent performance with 96.49% accuracy, 0.9942 AUC and 1.0000 precision. It has the highest MCC (0.9258) in this split and is competitive with Logistic Regression. |
| Overall Winner | **Random Forest** based on the highest MCC (0.9258) and perfect malignant precision (1.0000). Logistic Regression is an extremely close alternative and has the highest AUC (0.9960). |

### Interpretation of the Evaluation Metrics

- **Accuracy:** Proportion of all test samples classified correctly.
- **AUC:** Area under the ROC curve; measures ranking/discrimination ability
  across classification thresholds.
- **Precision:** Among samples predicted malignant, the proportion actually
  malignant.
- **Recall:** Among actual malignant samples, the proportion correctly
  identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Correlation between actual and predicted binary classes; values
  closer to 1 indicate stronger classification quality.

### Streamlit Application Features

The Streamlit application provides:

1. CSV test-data upload.
2. Model-selection dropdown.
3. Evaluation metrics for the selected model.
4. Confusion matrix.
5. Classification report.
6. Prediction results.
7. Download option for predictions.

### Streamlit Community Cloud Link

**Live App:** [PASTE YOUR STREAMLIT APP LINK HERE]

### Repository Structure

```text
ML_Assignment_2_WDBC/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

### BITS Virtual Lab Screenshot

**Insert ONE screenshot of the assignment execution on BITS Virtual Lab here.**

The assignment requires one screenshot as proof that the work was performed
on BITS Virtual Lab.

### Deployment Procedure

1. Create a GitHub repository.
2. Upload all files from this project folder.
3. Ensure `requirements.txt` is present in the repository root.
4. Open Streamlit Community Cloud.
5. Connect/authorize GitHub.
6. Create a new app.
7. Select the GitHub repository, branch `main`, and `app.py`.
8. Deploy the application.
9. Open the generated `streamlit.app` URL and test it with `test_data.csv`.
10. Copy the GitHub URL and Streamlit URL into this README/submission PDF.
11. Take the required BITS Virtual Lab screenshot.
12. Export the final README/submission document as a single PDF.

### Academic Integrity Note

The numerical results in this README correspond to the exact experimental
configuration stated above. The student should review the code, understand
the preprocessing, model choices and evaluation metrics, and make any
personal/custom changes required before submission.
