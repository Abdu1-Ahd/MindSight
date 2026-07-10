"""Test suite for MindSight data loading and preprocessing."""
import glob
import os

import numpy as np
import pandas as pd
import pytest

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'Track_D_Mental_Health'
)

T16 = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
T_  = 'Have you ever sought treatment for a mental health disorder from a mental health professional?'


def load_and_merge():
    """Load and merge all 7 OSMI CSVs. Mirrors Phase 1 logic."""
    dfs = []
    for f in sorted(glob.glob(os.path.join(DATA_PATH, '*.csv'))):
        yr = int(os.path.basename(f).replace('Mental_Health_Survey_', '').replace('.csv', ''))
        frame = pd.read_csv(f, low_memory=False)
        frame['year'] = yr
        for c in [T16, T_]:
            if c in frame.columns:
                frame = frame.rename(columns={c: 'treatment'})
                break
        dfs.append(frame)
    return pd.concat(dfs, ignore_index=True)


class TestDataLoading:
    """Verify CSV merge produces expected shape and columns."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_and_merge()

    def test_csv_files_exist(self):
        files = glob.glob(os.path.join(DATA_PATH, '*.csv'))
        assert len(files) == 7, f"Expected 7 CSVs, found {len(files)}"

    def test_merged_row_count(self):
        assert len(self.df) > 3000, f"Expected >3000 rows, got {len(self.df)}"

    def test_treatment_column_exists(self):
        assert 'treatment' in self.df.columns

    def test_year_column_exists(self):
        assert 'year' in self.df.columns

    def test_year_range(self):
        years = sorted(self.df['year'].unique())
        assert years == [2016, 2017, 2018, 2019, 2020, 2021, 2022]

    def test_no_empty_dataframes(self):
        for f in sorted(glob.glob(os.path.join(DATA_PATH, '*.csv'))):
            frame = pd.read_csv(f, low_memory=False)
            assert len(frame) > 0, f"{f} is empty"


class TestPreprocessing:
    """Verify preprocessing produces clean, model-ready data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        df = load_and_merge()
        if df['treatment'].dtype == object:
            df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0}).fillna(df['treatment'])
        df['treatment'] = pd.to_numeric(df['treatment'], errors='coerce')
        df = df.dropna(subset=['treatment'])
        df['treatment'] = df['treatment'].astype(int)

        X = df.drop(columns=['treatment', 'year']).copy()
        miss_rate = X.isnull().mean()
        X = X[miss_rate[miss_rate <= 0.5].index]

        HIGH_CARD = 50
        X = X[[c for c in X.columns if X[c].nunique() <= HIGH_CARD]]

        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        for c in cat_cols:
            X[c] = X[c].fillna('Unknown')
        for c in num_cols:
            X[c] = X[c].fillna(X[c].median())

        for c in cat_cols:
            uniq = sorted(X[c].unique(), key=str)
            mapping = {val: idx for idx, val in enumerate(uniq)}
            X[c] = X[c].map(mapping)

        self.X = X
        self.y = df.loc[X.index, 'treatment']

    def test_no_nan_after_cleaning(self):
        assert self.X.isnull().sum().sum() == 0, "NaN values remain after preprocessing"

    def test_target_binary(self):
        unique = set(self.y.unique())
        assert unique == {0, 1}, f"Target should be binary {{0, 1}}, got {unique}"

    def test_feature_count(self):
        assert self.X.shape[1] >= 10, f"Expected >=10 features, got {self.X.shape[1]}"

    def test_class_balance_not_degenerate(self):
        pos_rate = self.y.mean()
        assert 0.3 < pos_rate < 0.8, f"Class balance {pos_rate:.2%} is degenerate"


class TestModels:
    """Verify models produce accuracy above majority-class baseline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        df = load_and_merge()
        if df['treatment'].dtype == object:
            df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0}).fillna(df['treatment'])
        df['treatment'] = pd.to_numeric(df['treatment'], errors='coerce')
        df = df.dropna(subset=['treatment'])
        df['treatment'] = df['treatment'].astype(int)

        X = df.drop(columns=['treatment', 'year']).copy()
        miss_rate = X.isnull().mean()
        X = X[miss_rate[miss_rate <= 0.5].index]
        X = X[[c for c in X.columns if X[c].nunique() <= 50]]

        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        for c in cat_cols:
            X[c] = X[c].fillna('Unknown')
        for c in num_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in cat_cols:
            uniq = sorted(X[c].unique(), key=str)
            mapping = {val: idx for idx, val in enumerate(uniq)}
            X[c] = X[c].map(mapping)
        for c in X.columns:
            X[c] = X[c].astype(float)
            mu, sigma = X[c].mean(), X[c].std()
            X[c] = (X[c] - mu) / (sigma + 1e-8)

        X_np = X.values.astype(np.float64)
        y_np = df.loc[X.index, 'treatment'].values.astype(np.float64)

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X_np))
        split = int(len(X_np) * 0.8)
        self.X_train, self.X_test = X_np[idx[:split]], X_np[idx[split:]]
        self.y_train, self.y_test = y_np[idx[:split]], y_np[idx[split:]]
        self.baseline = max(self.y_test.mean(), 1 - self.y_test.mean())

    def test_perceptron_above_baseline(self):
        w = np.zeros(self.X_train.shape[1])
        b = 0.0
        for _ in range(50):
            for i in range(len(self.X_train)):
                pred = 1 if np.dot(w, self.X_train[i]) + b > 0 else 0
                err = self.y_train[i] - pred
                w += 0.1 * err * self.X_train[i]
                b += 0.1 * err
        preds = (self.X_test @ w + b > 0).astype(int)
        acc = (preds == self.y_test).mean()
        assert acc > self.baseline, f"Perceptron {acc:.4f} <= baseline {self.baseline:.4f}"

    def test_delta_rule_above_baseline(self):
        np.random.seed(42)
        w = np.random.randn(self.X_train.shape[1]) * 0.01
        b = 0.0
        for _ in range(100):
            y_hat = self.X_train @ w + b
            err = self.y_train - y_hat
            n = len(self.y_train)
            w -= 0.01 * (-(2 / n) * (self.X_train.T @ err))
            b -= 0.01 * (-(2 / n) * err.sum())
        preds = (self.X_test @ w + b > 0.5).astype(int)
        acc = (preds == self.y_test).mean()
        assert acc > self.baseline, f"Delta Rule {acc:.4f} <= baseline {self.baseline:.4f}"

    def test_kmeans_converges(self):
        K = 2
        rng = np.random.RandomState(42)
        centroids = self.X_train[rng.choice(len(self.X_train), K, replace=False)].copy()
        for it in range(100):
            dists = np.linalg.norm(self.X_train[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c = np.array([self.X_train[labels == k].mean(axis=0) for k in range(K)])
            if np.allclose(centroids, new_c):
                break
            centroids = new_c
        assert it < 99, "K-Means did not converge within 100 iterations"

# session:24b485fc
