# 🎬 Hollywood Movie Rating Predictor

A neural network regression model, built with **TensorFlow/Keras**, that predicts a Hollywood movie's rating (on a 1–10 scale) from structured production and release attributes such as budget, box-office revenue, genre, runtime, and director experience.

---

## 📌 Overview

This project trains a small feed-forward neural network (a Multi-Layer Perceptron) to estimate a movie's rating from 7 numeric input features. It's an end-to-end regression pipeline — load data, build the model, train it, and evaluate it with standard regression metrics.

> ⚠️ **Note on dataset size:** the dataset contains only 64 movie records, generated synthetically for learning purposes. With this little data, results are best understood as a demonstration of the ML workflow rather than a production-ready predictor — see [Limitations](#-limitations--future-work) below.

---

## 🗂️ Project Structure

```
Hollywoodmovie_Rating_Model/
│
├── Data.csv                              # Dataset (64 movie records, 7 features + target)
├── Hollywoodmovie_Rating_Model.ipynb     # Data loading, model building, training & evaluation
└── README.md                             # Project documentation (this file)
```

---

## 📊 Dataset

- **File:** `Data.csv`
- **Size:** 64 rows, 8 columns (7 features + 1 target)
- **Target column:** `Rating` (continuous, roughly 1–10)
- Categorical columns (`genre`, `release_season`) are already stored as integer codes in the raw file, so no additional encoding step is applied before training.

| Feature | Type | Description |
|---|---|---|
| `budget_million` | Numeric | Production budget (million USD) |
| `worldwide_gross_million` | Numeric | Worldwide box-office revenue (million USD) |
| `genre` | Numeric (encoded) | Integer code representing the movie's genre |
| `director_experience` | Numeric | Number of prior films / years of experience |
| `runtime_min` | Numeric | Movie runtime in minutes |
| `release_season` | Numeric (encoded) | Integer code representing the release season |
| `sequel` | Numeric (binary) | 1 if the movie is a sequel, 0 otherwise |
| `Rating` | Numeric (target) | The movie's rating — what the model predicts |

---

## 🧠 Model Architecture

The model is a small **feed-forward neural network** built with `keras.Sequential`:

```
Dense(7, activation="relu", input_shape=[7])
Dense(7, activation="relu")
Dense(1, activation="relu")   # output layer
```

- **Input:** all 7 raw features, fed directly into the network (no scaling/normalization is applied).
- **Optimizer:** Adam
- **Loss function:** Mean Squared Error (MSE) — the standard loss for regression problems.
- **Training:** 500 epochs on the full training split.
- **Train/test split:** a single 80/20 split via `train_test_split` (no cross-validation).

---

## 📈 Evaluation — and what the metrics mean

Since `Rating` is a continuous number, this is a **regression problem**, so performance is measured by how far predictions land from the true ratings — not by exact matches.

- **MAE (Mean Absolute Error):** the average size of the prediction error, ignoring direction. An MAE of 0.48 means predictions are off by about 0.48 rating points on average — e.g. predicting 7.5 for a movie actually rated 7.0 or 8.0.
- **RMSE (Root Mean Squared Error):** like MAE, but squares errors before averaging, so larger mistakes count more heavily. RMSE (0.56) being only slightly above MAE (0.48) suggests errors are fairly consistent, without major outlier predictions.
- **R² Score:** how much of the variation in ratings the model explains, from 0 (no better than predicting the average) to 1 (perfect). An R² of 0.70 means the model explains about **70% of the variance** in ratings — a solid result given the tiny dataset.

### ✅ Results

| Metric | Value | Meaning |
|---|---|---|
| MAE | **0.48** | Predictions are off by ~0.48 points on average |
| RMSE | **0.56** | Errors are fairly consistent, no large outliers |
| R² Score | **0.70** | Model explains ~70% of the variance in ratings |

---

## 🚀 How to Use

1. Clone or download the repository.
2. Open `Hollywoodmovie_Rating_Model.ipynb` in Jupyter or Google Colab.
3. Run the cells in order: load `Data.csv`, build the model, train it, and evaluate it on the held-out test split.
4. To predict the rating of a new movie, build a single-row DataFrame with the same 7 feature columns, in the same order, and pass it to `model.predict()`.

---

## ⚠️ Limitations & Future Work

- **Small, synthetic dataset (64 rows):** results can shift noticeably depending on how the data happens to be split into train/test. A larger, real-world dataset (e.g. from IMDb or TMDb) would give far more reliable results.
- **No feature scaling:** numeric features span very different ranges (e.g. budget in the hundreds vs. `sequel` as 0/1); normalizing inputs would likely help training stability and accuracy.
- **No cross-validation:** a single train/test split can be optimistic or pessimistic by chance; k-fold cross-validation would give a more trustworthy estimate of performance on such a small dataset.
- **ReLU activation on the output layer:** this clips any negative predictions to zero, which isn't strictly necessary for this target range; a linear output activation is more standard for regression.
- **Ideas for improvement:**
  - Collect more real-world data
  - Add feature scaling/normalization and k-fold cross-validation
  - Compare against simpler baselines (Linear Regression, Random Forest) to check whether the neural network is actually adding value
  - Add richer features (cast popularity, critic reviews, awards)
  - Add regularization (Dropout, early stopping) once more data is available

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

Dataset generated synthetically for educational/demo purposes.
