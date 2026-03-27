"""
音频分类训练器
基于 Mel 频谱图 + CNN，支持 CPU 训练
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Callable, Optional


class AudioClassifierTrainer:
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(
        self,
        feature_type: str = "mel_spectrogram",
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 64,
        n_mfcc: int = 40,
        output_dir: str = "./models",
    ):
        self.feature_type = feature_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.output_dir = output_dir

    def _load_audio_files(self, folder: str):
        """扫描文件夹，按子目录名作为类别"""
        classes = sorted([d for d in os.listdir(folder)
                          if os.path.isdir(os.path.join(folder, d))])
        if len(classes) < 2:
            raise ValueError(f"至少需要 2 个类别文件夹，当前: {classes}")

        files = []
        labels = []
        for ci, cls_name in enumerate(classes):
            cls_dir = os.path.join(folder, cls_name)
            for f in os.listdir(cls_dir):
                if os.path.splitext(f)[1].lower() in self.AUDIO_EXTS:
                    files.append(os.path.join(cls_dir, f))
                    labels.append(ci)

        return files, labels, classes

    def _extract_feature(self, filepath: str):
        """提取单个音频文件的特征（Mel 频谱图或 MFCC）"""
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(filepath)

        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 固定长度
        target_len = int(self.sample_rate * self.duration)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # 提取特征
        if self.feature_type == "mel_spectrogram":
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels
            )
            feature = transform(waveform)
            feature = torchaudio.transforms.AmplitudeToDB()(feature)
        else:  # mfcc
            transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate, n_mfcc=self.n_mfcc
            )
            feature = transform(waveform)

        return feature

    def train(self, folder: str, progress_callback: Optional[Callable] = None) -> dict:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset, random_split
        except ImportError as e:
            raise ImportError(f"缺少依赖：{e}\n请运行: pip install torch torchaudio") from e

        def _progress(pct, desc=""):
            if progress_callback:
                progress_callback(pct, desc=desc)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载数据
        _progress(0.05, "📂 扫描音频文件...")
        files, labels, classes = self._load_audio_files(folder)
        num_classes = len(classes)

        _progress(0.1, f"🎵 提取特征 (共 {len(files)} 个文件)...")
        features = []
        for i, f in enumerate(files):
            try:
                feat = self._extract_feature(f)
                features.append(feat)
            except Exception:
                continue
            if i % 10 == 0:
                _progress(0.1 + 0.2 * i / len(files), f"🎵 提取特征 {i+1}/{len(files)}")

        # 对齐特征尺寸
        min_time = min(f.shape[-1] for f in features)
        features = [f[..., :min_time] for f in features]
        valid_labels = labels[:len(features)]

        X = torch.stack(features)
        y = torch.tensor(valid_labels, dtype=torch.long)

        # 2. 划分数据集
        dataset = TensorDataset(X, y)
        val_size = max(int(len(dataset) * 0.2), num_classes)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2)

        # 3. 构建简单 CNN 模型
        _progress(0.35, "🏗️ 构建 CNN 模型...")
        in_channels = X.shape[1]
        freq_dim = X.shape[2]

        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # 4. 训练
        best_acc = 0.0
        best_epoch = 0
        best_state = None
        history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            _progress(0.35 + 0.55 * epoch / self.epochs, f"🔁 训练第 {epoch+1}/{self.epochs} 轮...")

            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_correct += (out.argmax(1) == yb).sum().item()
                train_total += xb.size(0)

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_total += xb.size(0)

            scheduler.step()

            t_acc = train_correct / train_total
            v_acc = val_correct / val_total
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)
            history["train_loss"].append(train_loss / train_total)
            history["val_loss"].append(val_loss / val_total)

            if v_acc > best_acc:
                best_acc = v_acc
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 5. 保存
        _progress(0.92, "💾 保存模型...")
        os.makedirs(self.output_dir, exist_ok=True)
        run_name = f"audio-cnn-{datetime.now().strftime('%m%d-%H%M')}"
        save_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            "model_state_dict": best_state,
            "num_classes": num_classes,
            "classes": classes,
            "feature_type": self.feature_type,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "n_mels": self.n_mels,
            "n_mfcc": self.n_mfcc,
            "in_channels": in_channels,
        }, os.path.join(save_dir, "model.pt"))

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({
                "classes": classes, "best_acc": best_acc,
                "best_epoch": best_epoch, "feature_type": self.feature_type,
                "param_count": param_count,
            }, f, ensure_ascii=False, indent=2)

        _progress(1.0, "✅ 训练完成！")

        return {
            "model_path": save_dir,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "num_classes": num_classes,
            "classes": classes,
            "train_size": train_size,
            "eval_size": val_size,
            "param_count": param_count,
            "history": history,
        }
