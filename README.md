# Titanic Survival Prediction

A beginner-friendly Machine Learning project that predicts whether a
Titanic passenger would survive based on passenger details. The project
includes preprocessing, model comparison, evaluation, and a Streamlit
web interface.

> **Disclaimer:** Predictions are model estimates based on historical
> data, not guarantees about real-world outcomes.

## Project Features

-   Handles missing numerical and categorical values.
-   Scales numerical features and one-hot encodes categorical features.
-   Compares seven classification models.
-   Evaluates models using Accuracy, Precision, Recall, F1-score, and a
    Confusion Matrix.
-   Uses 5-fold cross-validation to compare selected models.
-   Provides an interactive Streamlit app for predictions.

## Dataset

The project uses the Titanic training dataset with 891 rows and 12
original columns.

-   **Target:** `Survived` (`0` = did not survive, `1` = survived)
-   **Input features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`,
    `Fare`, `Embarked`
-   **Dropped columns:** `PassengerId`, `Name`, `Ticket`, `Cabin`
-   **Missing values:** `Age` and `Embarked` are imputed in the
    preprocessing pipeline.

## Models Compared

The initial test-split comparison included:

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Decision Tree
-   Random Forest
-   Support Vector Machine (SVM)
-   Extra Trees
-   AdaBoost

### Initial test-split results

  Model                   Accuracy   Precision   Recall   F1-score
  --------------------- ---------- ----------- -------- ----------
  Decision Tree              0.816       0.790    0.710      0.748
  KNN                        0.816       0.800    0.696      0.744
  Random Forest              0.816       0.800    0.696      0.744
  Extra Trees                0.804       0.758    0.725      0.741
  SVM                        0.816       0.821    0.667      0.736
  Logistic Regression        0.804       0.793    0.667      0.724
  AdaBoost                   0.782       0.750    0.652      0.698

These are results from the initial test-split comparison. Scores can
vary with the split and model configuration.

## Cross-validation

A 5-fold cross-validation comparison on the training data gave:

  Model                   Mean F1-score   Standard deviation
  --------------------- --------------- --------------------
  Logistic Regression             0.724                0.041
  Random Forest                   0.722                0.072
  Decision Tree                   0.708                0.054

Logistic Regression was selected as the final candidate because it had
the highest mean cross-validation F1-score among these three models. The
difference from Random Forest was small, so this should not be treated
as proof that it is universally the best model.

## Final test evaluation

The final Logistic Regression classification report on the 179-row test
set:

  Class                Precision   Recall   F1-score   Support
  ------------------ ----------- -------- ---------- ---------
  Did Not Survive           0.81     0.89       0.85       110
  Survived                  0.79     0.67       0.72        69
  **Accuracy**                              **0.80**   **179**
  Macro average             0.80     0.78       0.79       179
  Weighted average          0.80     0.80       0.80       179

Confusion matrix (rows = actual class, columns = predicted class):

                              Predicted: Did Not Survive   Predicted: Survived
  ------------------------- ---------------------------- ---------------------
  Actual: Did Not Survive                             98                    12
  Actual: Survived                                    23                    46

The model correctly classified 144 out of 179 test passengers (about
80.45%). It identified 46 of the 69 actual survivors; recall for the
`Survived` class was about 66.7%, so improving survivor recall is a
possible next step.

## Project Structure

Keep these files in the same folder:

``` text
Titanic_Project/
├── app.py
├── titanic_survival_model.pkl
└── README.md
```

-   `app.py` --- Streamlit interface.
-   `titanic_survival_model.pkl` --- saved fitted
    preprocessing-and-model pipeline.
-   `README.md` --- project documentation.

If you also keep the notebook or dataset in the folder, add those files
to this list.

## Installation

Use Python in an environment where the required packages are installed:

``` bash
pip install streamlit pandas scikit-learn joblib
```

## Run the app

Open a terminal in the project folder and run:

``` bash
streamlit run app.py
```

Streamlit will provide a local URL to open the app in your browser.

## Notes

-   The saved `.pkl` file should be created from the same preprocessing
    pipeline and final model used for evaluation.
-   Keep `app.py` and `titanic_survival_model.pkl` in the same directory
    unless the model path in the app is updated.
-   The app's displayed survival probability is the model's estimated
    probability, not a guaranteed outcome.
-   This is an educational portfolio project using a historical dataset.
    It should not be used for real-world safety decisions.

## Possible Improvements

-   Compare survivor recall and F1-score across models.
-   Tune model hyperparameters using cross-validation.
-   Add a precision-recall curve and feature importance/interpretability
    analysis where appropriate.
-   Add screenshots of the Streamlit app and confusion matrix.
