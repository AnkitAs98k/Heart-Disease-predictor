# Heart Disease Risk Estimator — local frontend

A small Flask app + form frontend for the model in `ML1.ipynb`
(github.com/AnkitAs98k/Machine-learning). Enter clinical measurements,
get a risk estimate from a trained SVM.

## Setup

```bash
pip install -r requirements.txt
```

Put the dataset (`heart.csv`, the Kaggle "Heart Failure Prediction Dataset")
in this folder, then train:

```bash
python train_model.py
```

This prints an accuracy/F1 comparison across Logistic Regression, Naive
Bayes, Decision Tree, SVM, and KNN — same models as the notebook — and
saves `model.pkl`, `scaler.pkl`, `feature_columns.json`.

Then run the app:

```bash
python app.py
```

Open **http://localhost:5000**.

## What was changed from the notebook

- **`ST_Slope` bug fixed.** In `ML1.ipynb`, cell 15 does
  `njdf_heart = pd.get_dummies(df_heart, columns=['ST_Slope'])` — the
  result is assigned to a new variable that's never used again, so
  `ST_Slope` stays a string column. Running the notebook straight
  through as saved breaks at `StandardScaler.fit_transform` because it
  can't scale text. `train_model.py` reassigns the dummies back to the
  dataframe.
- **`SVC(probability=True)`.** The notebook uses default `SVC()`, which
  can't return prediction probabilities. It's needed here to show a
  confidence percentage in the UI, so it's enabled (this makes training
  slightly slower via internal cross-validation, negligible on this
  dataset's size).
- **`heart.csv` and the exported `.pkl` files aren't in the repo** —
  only the notebook is committed, so this app expects you to supply the
  CSV and train locally rather than loading a pre-trained model.

## Files

```
app.py                 Flask server + /predict API
train_model.py          Fixed training pipeline
templates/index.html    Form UI
static/style.css        Styling
static/script.js        Form submit + result rendering
requirements.txt
```
