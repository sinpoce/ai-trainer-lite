"""
文本分类训练器
基于 HuggingFace Transformers + DistilBERT，支持 CPU 训练
"""

import os
import time
import numpy as np
from datetime import datetime
from typing import Callable, Optional
import pandas as pd


class TextClassifierTrainer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        max_length: int = 128,
        warmup_ratio: float = 0.1,
        output_dir: str = "./models",
    ):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir

    def train(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        test_size: float = 0.2,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score, f1_score
            from torch.utils.data import Dataset
        except ImportError as e:
            raise ImportError(f"缺少依赖：{e}\n请运行: pip install transformers torch scikit-learn") from e

        def _progress(pct, desc=""):
            if progress_callback:
                progress_callback(pct, desc=desc)

        # ── 1. 数据准备 ────────────────────────────────────────────────
        _progress(0.15, desc="🔤 编码标签...")
        df = df.dropna(subset=[text_col, label_col]).copy()
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)

        le = LabelEncoder()
        df["label_id"] = le.fit_transform(df[label_col])
        classes = list(le.classes_)
        num_classes = len(classes)
        id2label = {i: c for i, c in enumerate(classes)}
        label2id = {c: i for i, c in enumerate(classes)}

        train_df, eval_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label_id"])

        # ── 2. Tokenizer ───────────────────────────────────────────────
        _progress(0.2, desc="⬇️ 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    list(texts), truncation=True, padding=True,
                    max_length=max_length, return_tensors="pt"
                )
                self.labels = torch.tensor(list(labels), dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = self.labels[idx]
                return item

        train_dataset = TextDataset(train_df[text_col], train_df["label_id"], tokenizer, self.max_length)
        eval_dataset = TextDataset(eval_df[text_col], eval_df["label_id"], tokenizer, self.max_length)

        # ── 3. 模型 ────────────────────────────────────────────────────
        _progress(0.35, desc="🏗️ 加载预训练模型...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )

        # ── 4. 训练配置 ────────────────────────────────────────────────
        run_name = f"{self.model_name.split('/')[-1]}-{datetime.now().strftime('%m%d-%H%M')}"
        model_save_path = os.path.join(self.output_dir, run_name)
        os.makedirs(model_save_path, exist_ok=True)

        total_steps = (len(train_dataset) // self.batch_size) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted"),
            }

        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.lr,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=10,
            report_to="none",
            no_cuda=not torch.cuda.is_available(),
            dataloader_num_workers=0,
        )

        _progress(0.4, desc=f"🚀 开始训练（{self.epochs} 轮，CPU 模式）...")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        train_result = trainer.train()
        _progress(0.9, desc="💾 保存模型...")

        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # ── 5. 评估 ────────────────────────────────────────────────────
        eval_result = trainer.evaluate()
        _progress(1.0, desc="✅ 完成！")

        return {
            "model_path": model_save_path,
            "accuracy": eval_result["eval_accuracy"],
            "f1": eval_result["eval_f1"],
            "total_samples": len(df),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "num_classes": num_classes,
            "classes": classes,
            "training_time": train_result.metrics.get("train_runtime", 0),
            "plot": None,
        }
