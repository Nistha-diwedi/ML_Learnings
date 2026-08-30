# ML — Machine Learning Notebooks Collection

A hands-on collection of Jupyter notebooks covering core Machine Learning concepts — from unsupervised clustering and dimensionality reduction to ensemble methods, hyperparameter tuning, NLP, and statistical hypothesis testing. All notebooks use synthetic or classic datasets (`sklearn`, `seaborn`) and a text emotion dataset (`train.txt`).

> **Scope:** All files for this project live inside `C:\Users\DELL\Downloads\ML\`. No files outside this folder are required or modified.

---

## 📁 Project Structure

```
ML/
├── Clustering.ipynb                  # K-Means, Elbow Method, DBSCAN (make_blobs / make_moons)
├── PCA.ipynb                         # Principal Component Analysis (5D → 2D visualization)
├── Ensemble_Learning_Methods.ipynb   # Stacking, Bagging, Boosting (AdaBoost, GradientBoost, XGBoost)
├── Hyperparameter_Methods.ipynb      # GridSearchCV & RandomizedSearchCV (KNN, SVM on Iris)
├── NLP.ipynb                         # Text preprocessing + BoW / TF-IDF + Naive Bayes & Logistic Regression
├── z,_t,_2_sample_tests.ipynb        # Z-test, T-test, Two-Sample T-test (scipy.stats)
├── train.txt                         # Emotion classification dataset (16000 rows, text;emotion)
└── README.md                         # This file
```

---

## 📊 Dataset — `train.txt`

- **Format:** `text;emotion` (semicolon-separated, no header, loaded with `pd.read_csv(..., sep=';', names=['text','emotion'])`)
- **Rows:** 16,000
- **Example:**
  ```
  i didnt feel humiliated;sadness
  i am feeling grouchy;anger
  i feel romantic too;love
  ```
- **Labels (6 emotions):** `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
- **Usage:** Primary dataset for `NLP.ipynb`. Other notebooks use `sklearn.datasets.make_blobs`, `make_moons`, and `seaborn.load_dataset('iris')`.
- **Source:** Kaggle-style emotion dataset (included locally).

---

## 📓 Notebooks

### 1. `Clustering.ipynb` — 31 cells
**Topic:** Unsupervised Learning — K-Means Clustering
- Generate synthetic data with `make_blobs(n_samples=500, centers=3)`
- Standardization with `StandardScaler` (unsupervised — no `y` needed)
- **Elbow Method / WCSS (Inertia):** iterate `k=1..10`, plot inertia to select `k=3`
- Fit final `KMeans(n_clusters=3)` and visualize clusters
- **Non-circular clustering demo:** `make_moons` + comparison of `KMeans` vs `DBSCAN` to show K-Means failure on non-spherical data

### 2. `PCA.ipynb` — 7 cells
**Topic:** Dimensionality Reduction — Principal Component Analysis
- Generate `make_blobs(n_samples=500, n_features=5, centers=3)`
- Standardize with `StandardScaler`
- `PCA(n_components=2)` → project 5D → 2D (`PC1`, `PC2`)
- Scatter plot with `seaborn.scatterplot` colored by cluster label

### 3. `Ensemble_Learning_Methods.ipynb` — 39 cells
**Topic:** Ensemble Learning on Iris dataset (`sns.load_dataset('iris')`)
- **Stacking:** `StackingClassifier` with base learners `DecisionTreeClassifier`, `SVC(rbf)`, `LogisticRegression` + meta-learner `LogisticRegression`, `cv=5`
- **Bagging:** `RandomForestClassifier` (with `cross_val_score`)
- **Boosting:**
  - `AdaBoostClassifier`
  - `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)`
  - `XGBClassifier` (`xgboost`)
- Metrics: `accuracy_score`, `classification_report`, stratified `train_test_split`

### 4. `Hyperparameter_Methods.ipynb` — 31 cells
**Topic:** Hyperparameter Tuning on Iris
- Baseline models: `KNeighborsClassifier(n_neighbors=5)` and `SVC(gamma='auto')`
- Demonstrates how changing `C`, `kernel`, `gamma` affects accuracy (including overfitting to 100%)
- **GridSearchCV:** `GridSearchCV(SVC(gamma='auto'), {'C':[1,10,20,30], 'kernel':['rbf','linear']}, cv=5)`
- **RandomizedSearchCV:** `RandomizedSearchCV(..., n_iter=4, cv=5)` — discussion of when random search is preferred (large search spaces like RandomForest/XGBoost)

### 5. `NLP.ipynb` — 45 cells
**Topic:** Natural Language Processing — Emotion Classification
- Load `train.txt` → map emotion strings to integers
- **Preprocessing pipeline:**
  1. Lowercasing
  2. Punctuation removal (`string.punctuation`)
  3. Number removal
  4. URL removal
  5. Emoji / non-ASCII removal (`isascii()`)
  6. Stopword removal (`nltk`: `punkt`, `stopwords`, `punkt_tab`, `word_tokenize`)
  7. (Notes on HTML tag handling — not needed for this dataset)
- **Vectorization demos:**
  - Bag-of-Words with `CountVectorizer` (including `ngram_range` option)
  - TF-IDF with `TfidfVectorizer`
- **Classification:**
  - `train_test_split(X, y, test_size=0.33)`
  - `MultinomialNB` + BoW and TF-IDF
  - `LogisticRegression(max_iter=1000)` + TF-IDF
  - Accuracy comparison

### 6. `z,_t,_2_sample_tests.ipynb` — 17 cells
**Topic:** Statistical Hypothesis Testing (`scipy.stats`)
- **Z-test:** `z = (sample_mean - pop_mean) / (pop_std / sqrt(n))`, two-tailed `p = 2*(1 - norm.cdf(|z|))`, `alpha=0.05`
- **T-test (One-Sample):** `t = (sample_mean - pop_mean) / (sample_std / sqrt(n))` with `ddof=1`, `scipy.stats.ttest_1samp`
- **Two-Sample T-test (Welch's):** `stats.ttest_ind(grp_a, grp_b, equal_var=False)` on synthetic groups (e.g., exam scores)
- Sample data: `n=10` heights around `pop_mean=170, pop_std=3`

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Language | Python 3 |
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn` (`KMeans`, `DBSCAN`, `PCA`, `SVC`, `KNN`, `RandomForest`, `AdaBoost`, `GradientBoosting`, `StackingClassifier`, `GridSearchCV`, `RandomizedSearchCV`, `TfidfVectorizer`, `CountVectorizer`), `xgboost` |
| NLP | `nltk`, `re`, `string` |
| Stats | `scipy.stats` (`norm`, `ttest_ind`, `ttest_1samp`) |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy nltk
```

Download NLTK data (first run of `NLP.ipynb` handles this):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### Run
1. Clone or copy the `ML/` folder locally (keep `train.txt` alongside `NLP.ipynb`).
2. Launch Jupyter:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```
3. Open any `.ipynb` in order — notebooks are self-contained and generate or load their own data.
4. For `NLP.ipynb`, ensure `train.txt` is in the same directory (already included).

> Note: The file `z,_t,_2_sample_tests.ipynb` contains a comma in its name. On Windows this is valid; if you move the project to a system that disallows commas, rename it to `z_t_2_sample_tests.ipynb` and update references.

---

## 🎯 Learning Outcomes

- When and why to use unsupervised vs supervised methods
- How to select `k` in K-Means and when to prefer density-based clustering (DBSCAN)
- How PCA preserves variance while reducing dimensions for visualization
- Trade-offs between stacking, bagging, and boosting ensembles
- Systematic hyperparameter optimization and avoiding overfitting via cross-validation
- End-to-end NLP pipeline: cleaning → vectorization → classification
- Interpreting Z / T / two-sample tests and p-values for hypothesis testing

---

## 📄 License & Notes

- This is an educational / portfolio project. Datasets (`iris`, synthetic blobs/moons, `train.txt` emotion data) are used for learning purposes.
- All notebooks use `random_state=42` for reproducibility where applicable.
- No external data download is required — everything runs offline with the included `train.txt`.

---

*Created: August 2026 · Location: `C:\Users\DELL\Downloads\ML\`*
