"""
表格数据训练器
自动尝试多种算法，找到最适合的模型（AutoML 风格）
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Optional, List


CLASSIFIERS = {
    "LogisticRegression": {
        "cls": "sklearn.linear_model.LogisticRegression",
        "params": {"max_iter": 1000, "random_state": 42},
    },
    "RandomForest": {
        "cls": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    },
    "GradientBoosting": {
        "cls": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {"n_estimators": 100, "random_state": 42},
    },
    "XGBoost": {
        "cls": "xgboost.XGBClassifier",
        "params": {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss", "verbosity": 0},
    },
    "SVM": {
        "cls": "sklearn.svm.SVC",
        "params": {"random_state": 42, "probability": True},
    },
    "KNN": {
        "cls": "sklearn.neighbors.KNeighborsClassifier",
        "params": {"n_neighbors": 5},
    },
    "NaiveBayes": {
        "cls": "sklearn.naive_bayes.GaussianNB",
        "params": {},
    },
}

REGRESSORS = {
    "LinearRegression": {
        "cls": "sklearn.linear_model.LinearRegression",
        "params": {},
    },
    "Ridge": {
        "cls": "sklearn.linear_model.Ridge",
        "params": {"random_state": 42},
    },
    "RandomForest": {
        "cls": "sklearn.ensemble.RandomForestRegressor",
        "params": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    },
    "GradientBoosting": {
        "cls": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {"n_estimators": 100, "random_state": 42},
    },
    "XGBoost": {
        "cls": "xgboost.XGBRegressor",
        "params": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
    },
    "SVR": {
        "cls": "sklearn.svm.SVR",
        "params": {},
    },
}


def _import_class(dotted_path: str):
    """动态导入类"""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TabularTrainer:
    def __init__(self, task_type: str = "classification", algorithms: List[str] = None):
        self.task_type = task_type
        self.algorithms = algorithms or ["all"]
        self.output_dir = "./models"

    def _build_preprocessor(self, df: pd.DataFrame, feature_cols: list):
        """构建数据预处理管道"""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
        from sklearn.impute import SimpleImputer

        num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if num_cols:
            num_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, num_cols))

        if cat_cols:
            cat_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("cat", cat_pipeline, cat_cols))

        if not transformers:
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer()

        return ColumnTransformer(transformers)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, r2_score
        except ImportError as e:
            raise ImportError(f"缺少依赖：{e}\n请运行: pip install scikit-learn") from e

        def _progress(pct, desc=""):
            if progress_callback:
                progress_callback(pct, desc=desc)

        # ── 1. 数据准备 ────────────────────────────────────────────────
        _progress(0.05, desc="🔍 分析特征...")
        df = df.dropna(subset=[target_col]).copy()
        feature_cols = [c for c in df.columns if c != target_col]

        y = df[target_col]
        if self.task_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            classes = list(le.classes_)
        else:
            y = y.astype(float)
            classes = []

        X = df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if self.task_type == "classification" else None
        )

        preprocessor = self._build_preprocessor(df, feature_cols)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        # ── 2. 选择算法 ────────────────────────────────────────────────
        algo_pool = CLASSIFIERS if self.task_type == "classification" else REGRESSORS
        if "all" in self.algorithms:
            selected = algo_pool
        else:
            selected = {k: v for k, v in algo_pool.items() if k in self.algorithms}

        # ── 3. 训练所有算法 ────────────────────────────────────────────
        all_results = {}
        best_score = -np.inf
        best_model = None
        best_algo = None

        total = len(selected)
        for i, (name, config) in enumerate(selected.items()):
            _progress(0.15 + 0.7 * i / total, desc=f"🔁 训练 {name}...")
            try:
                ModelClass = _import_class(config["cls"])
                model = ModelClass(**config["params"])
                t0 = time.time()
                model.fit(X_train_t, y_train)
                elapsed = time.time() - t0

                if self.task_type == "classification":
                    score = accuracy_score(y_test, model.predict(X_test_t))
                else:
                    score = r2_score(y_test, model.predict(X_test_t))

                all_results[name] = {"score": score, "time": elapsed, "model": model}

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_algo = name

            except Exception as e:
                all_results[name] = {"score": 0, "time": 0, "error": str(e)}

        # ── 4. 特征重要性 ──────────────────────────────────────────────
        feature_importance = {}
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            # 获取转换后的特征名
            try:
                feat_names = preprocessor.get_feature_names_out()
            except Exception:
                feat_names = [f"f{i}" for i in range(len(importances))]
            feature_importance = dict(
                sorted(zip(feat_names, importances), key=lambda x: -x[1])
            )

        # ── 5. 保存模型 ────────────────────────────────────────────────
        _progress(0.9, desc="💾 保存最优模型...")
        os.makedirs(self.output_dir, exist_ok=True)
        model_name = f"tabular-{best_algo}-{datetime.now().strftime('%m%d-%H%M')}.pkl"
        model_path = os.path.join(self.output_dir, model_name)

        bundle = {
            "model": best_model,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "classes": classes if self.task_type == "classification" else [],
            "task_type": self.task_type,
            "best_algorithm": best_algo,
            "best_score": best_score,
        }
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)

        _progress(1.0, desc="✅ 完成！")

        return {
            "model_path": model_path,
            "best_algorithm": best_algo,
            "best_score": best_score,
            "all_results": {k: {"score": v["score"], "time": v.get("time", 0)} for k, v in all_results.items()},
            "feature_importance": feature_importance,
            "classes": classes,
            "plot": None,
        }
