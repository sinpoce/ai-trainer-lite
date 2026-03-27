"""
图像分类训练器
基于 PyTorch + 迁移学习，支持 MobileNet/ResNet/EfficientNet，CPU 可运行
"""

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Callable, Optional


class ImageClassifierTrainer:
    ARCH_MAP = {
        "mobilenet_v3_small (轻量 1.5M, 推荐CPU)": "mobilenet_v3_small",
        "resnet18 (经典 11M)": "resnet18",
        "efficientnet_b0 (高效 5.3M)": "efficientnet_b0",
        "vgg11 (经典 128M)": "vgg11",
    }

    def __init__(
        self,
        model_arch: str = "mobilenet_v3_small (轻量 1.5M, 推荐CPU)",
        epochs: int = 10,
        img_size: int = 224,
        batch_size: int = 16,
        lr: float = 1e-3,
        augment: bool = True,
        output_dir: str = "./models",
    ):
        self.arch = self.ARCH_MAP.get(model_arch, model_arch)
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.augment = augment
        self.output_dir = output_dir

    def _count_params(self, model) -> int:
        return sum(p.numel() for p in model.parameters())

    def train(self, folder: str, progress_callback: Optional[Callable] = None) -> dict:
        try:
            import torch
            import torch.nn as nn
            from torchvision import datasets, transforms, models
            from torch.utils.data import DataLoader, random_split
        except ImportError as e:
            raise ImportError(f"缺少依赖：{e}\n请运行: pip install torch torchvision") from e

        def _progress(pct, desc=""):
            if progress_callback:
                progress_callback(pct, desc=desc)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 1. 数据增强 ────────────────────────────────────────────────
        _progress(0.05, desc="📂 扫描数据集...")
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.augment:
            train_tf = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ])

        val_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        full_dataset = datasets.ImageFolder(folder)
        classes = full_dataset.classes
        num_classes = len(classes)

        if num_classes < 2:
            raise ValueError(f"需要至少 2 个类别文件夹，当前只有: {classes}")

        # ── 2. 数据集划分 ──────────────────────────────────────────────
        val_size = max(int(len(full_dataset) * 0.2), num_classes)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_dataset.dataset.transform = train_tf
        val_dataset.dataset.transform = val_tf

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        # ── 3. 加载预训练模型 ──────────────────────────────────────────
        _progress(0.2, desc=f"🏗️ 加载 {self.arch} 预训练权重...")
        model_fn = getattr(models, self.arch)
        model = model_fn(weights="DEFAULT")

        # 替换最后分类层
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型架构: {self.arch}")

        param_count = self._count_params(model)
        model = model.to(device)

        # ── 4. 训练配置 ────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # ── 5. 训练循环 ────────────────────────────────────────────────
        best_acc = 0.0
        best_loss = float("inf")
        best_epoch = 0
        history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            _progress(
                0.25 + 0.65 * epoch / self.epochs,
                desc=f"🔁 训练第 {epoch+1}/{self.epochs} 轮..."
            )

            # 训练
            model.train()
            train_loss, train_correct = 0.0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            # 验证
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            t_acc = train_correct / train_size
            v_acc = val_correct / val_size
            t_loss = train_loss / train_size
            v_loss = val_loss / val_size
            scheduler.step()

            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if v_acc > best_acc:
                best_acc = v_acc
                best_loss = v_loss
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── 6. 保存模型 ────────────────────────────────────────────────
        _progress(0.92, desc="💾 保存模型...")
        os.makedirs(self.output_dir, exist_ok=True)
        run_name = f"img-{self.arch}-{datetime.now().strftime('%m%d-%H%M')}"
        save_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            "model_state_dict": best_state,
            "arch": self.arch,
            "num_classes": num_classes,
            "classes": classes,
            "img_size": self.img_size,
        }, os.path.join(save_dir, "model.pt"))

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({
                "arch": self.arch, "num_classes": num_classes,
                "classes": classes, "img_size": self.img_size,
                "best_acc": best_acc, "best_epoch": best_epoch,
            }, f, ensure_ascii=False, indent=2)

        _progress(1.0, desc="✅ 训练完成！")

        return {
            "model_path": save_dir,
            "best_acc": best_acc,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "num_classes": num_classes,
            "classes": classes,
            "train_size": train_size,
            "eval_size": val_size,
            "param_count": param_count,
            "history": history,
            "device": str(device),
        }

    @classmethod
    def load(cls, model_dir: str):
        """加载已训练的模型用于推理"""
        import torch
        from torchvision import models, transforms
        import torch.nn as nn

        checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        arch = checkpoint["arch"]
        num_classes = checkpoint["num_classes"]
        classes = checkpoint["classes"]
        img_size = checkpoint["img_size"]

        model_fn = getattr(models, arch)
        model = model_fn(weights=None)

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        trainer = cls.__new__(cls)
        trainer.model = model
        trainer.classes = classes
        trainer.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return trainer

    def predict(self, img):
        """推理单张图片，返回 (class_index, confidence)"""
        import torch
        x = self.transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)
        return pred.item(), conf.item()
