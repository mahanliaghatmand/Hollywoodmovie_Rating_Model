# 🎬 Hollywood Movie Rating Predictor

A deep learning regression model that predicts a Hollywood movie's rating (on a 1–10 scale) based on production and release attributes such as budget, box-office revenue, genre, runtime, and more.

---

## 📌 Description

**Hollywood Movie Rating Predictor** is a neural network (built with TensorFlow/Keras) trained to estimate the rating a movie would receive, given a small set of structured features. The dataset is a synthetic 64-row CSV file generated for experimentation and learning purposes, where `Rating` is a continuous target value (e.g. 6.0, 7.2, 8.5).

> ⚠️ **Note on dataset size:** with only 64 samples, this project is best treated as a learning/demo exercise rather than a production-grade model. Results can vary significantly between runs, and the model is prone to overfitting. See the [Limitations](#-limitations--recommendations) section for tips.

---

## 🗂️ Project Structure

```
Hollywoodmovie_Rating_Model/
│
├── data.csv          # Dataset (64 movie records)
├── Hollywoodmovie_Rating_Model.ipynb    # EDA, preprocessing, model training & evaluation
└── README.md         # Project documentation (this file)
```

---

## 📊 Dataset

- **Source:** Synthetic dataset (generated with the help of Claude AI) for educational purposes
- **Size:** 64 rows
- **Target column:** `Rating` (float, typically between 1 and 10)

### Features

| Feature | Type | Description |
|---|---|---|
| `budget_million` | Numeric | Production budget (in million USD) |
| `worldwide_gross_million` | Numeric | Worldwide box-office revenue (in million USD) |
| `genre` | Categorical | Movie genre (e.g. Action, Drama, Comedy) |
| `director_experience` | Numeric | Years of experience / number of prior films by the director |
| `runtime_min` | Numeric | Movie runtime in minutes |
| `release_season` | Categorical | Season of release (e.g. Summer, Winter, Holiday) |
| `Rating` | Numeric (target) | The movie's rating (1–10) — what the model predicts |

---

## ⚙️ Preprocessing

Since `genre` and `release_season` are text-based (categorical) columns, they can't be fed into a neural network directly — they need to be converted into numbers first. This is typically done with **One-Hot Encoding**, which turns each category into its own 0/1 column (e.g. `genre_Action`, `genre_Drama`, ...), so the model can understand it without assuming a false numeric order between categories.

The numeric columns (`budget_million`, `worldwide_gross_million`, `director_experience`, `runtime_min`) should be **scaled/normalized** — meaning their values are rescaled to a similar range (e.g. 0 to 1, or mean 0). Neural networks train faster and more reliably when input features are on comparable scales; otherwise a feature like budget (in the hundreds) can dominate a feature like director experience (in single digits).

Finally, the data should be split into a **training set** and a **test set** — the model learns only from the training set, and the test set is held back to check how well it performs on data it has never seen. Given that there are only 64 rows total, it's better to use **cross-validation** (training and testing on different slices of the data multiple times and averaging the results) instead of a single train/test split, since one split alone can give a misleadingly optimistic or pessimistic result with so little data.

---

## 🧠 Model

The model is a **feed-forward neural network** (a basic deep learning architecture, sometimes called a Multi-Layer Perceptron) built with TensorFlow/Keras. It consists of a few stacked "dense" layers, each layer learning increasingly abstract combinations of the input features, narrowing down (e.g. 64 → 32 → 16 neurons) until it reaches a single output neuron — the predicted rating. Since this is a regression problem (predicting a continuous number, not a category), the final layer has **no activation function** (a linear output), and the model is trained to minimize the **Mean Squared Error** between its predictions and the actual ratings.

---

## 📈 Evaluation — and what the metrics mean

Since the target (`Rating`) is a continuous number rather than a fixed set of classes, this is a **regression problem**, not classification — so the right way to judge the model is by measuring *how far off* its predictions are from the true ratings, not whether it got an exact match.

- **MAE (Mean Absolute Error):** the average size of the prediction error, ignoring direction (over or under). A MAE of 0.48 means that, on average, the model's predicted rating is off by about 0.48 points from the true rating — for example, predicting 7.5 for a movie actually rated 7.0 or 8.0.

- **RMSE (Root Mean Squared Error):** similar to MAE, but it squares the errors before averaging, which makes larger mistakes count more heavily. If RMSE is noticeably higher than MAE, it usually means a few predictions were way off, even if most were close. Here, RMSE (0.56) is only slightly above MAE (0.48), which suggests the model's errors are fairly consistent — no major outlier mistakes.

- **R² Score (Coefficient of Determination):** measures how much of the variation in ratings the model is able to explain, on a scale where 1.0 means perfect prediction and 0 means the model is no better than just guessing the average rating every time. An R² of 0.70 means the model explains about **70% of the variation** in movie ratings based on the given features — a reasonably strong result, especially considering the dataset only has 64 movies.

### ✅ Current Results

| Metric | Value | Meaning |
|---|---|---|
| MAE | **0.48** | Predictions are off by ~0.48 points on average |
| RMSE | **0.56** | Confirms errors are fairly consistent (no big outliers) |
| R² Score | **0.70** | Model explains ~70% of the variance in ratings |

---

## 🚀 How to Use

1. Clone or download the repository.
2. Open `notebook.ipynb` in Jupyter and run the cells in order: loading the data, preprocessing it, training the model, and evaluating it.
3. To predict the rating of a new movie, its feature values need to go through the same encoding and scaling steps used during training before being passed to the model.

---

## ⚠️ Limitations & Recommendations

- **Small dataset (64 rows):** results can shift a lot depending on how the data happens to be split into train/test. Cross-validation gives a more trustworthy picture than a single split.
- **Synthetic data:** since the dataset was generated rather than collected from a real source (like IMDb or Kaggle), the model may not generalize well to real-world movies.
- **Ideas for improvement:**
  - Collect more real-world data (e.g. from IMDb or Kaggle)
  - Compare against simpler baseline models (Linear Regression, Random Forest) to see if the deep learning model is actually adding value
  - Add regularization (Dropout) and early stopping to reduce the risk of overfitting on such a small dataset
  - Add richer features (cast popularity, critic reviews, awards, etc.)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

Dataset generated with the assistance of Claude AI for educational/demo purposes.
