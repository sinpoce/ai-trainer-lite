"""
ONNX 模型导出工具
将训练好的模型转换为 ONNX 格式，支持跨平台部署
"""

import os
import json
import pickle
from typing import Optional


def export_text_model_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
) -> str:
    """导出 HuggingFace 文本分类模型为 ONNX"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError("需要安装 torch 和 transformers")

    if output_path is None:
        output_path = os.path.join(model_path, "model.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dummy_input = tokenizer(
        "This is a sample text for export",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    torch.onnx.export(
        model,
        tuple(dummy_input.values()),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # 保存 label 映射
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        label_map = config.get("id2label", {})
        meta_path = os.path.join(os.path.dirname(output_path), "onnx_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"id2label": label_map, "model_type": "text_classification"}, f, indent=2)

    return output_path


def export_tabular_model_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
) -> str:
    """导出 scikit-learn 表格模型为 ONNX"""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np
    except ImportError:
        raise ImportError("需要安装 skl2onnx: pip install skl2onnx")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    n_features = len(feature_cols)

    if output_path is None:
        output_path = model_path.replace(".pkl", ".onnx")

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 保存元数据
    meta_path = output_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "classes": bundle.get("classes", []),
            "task_type": bundle.get("task_type", "classification"),
            "model_type": "tabular",
        }, f, indent=2, ensure_ascii=False)

    return output_path


def export_image_model_to_onnx(
    model_dir: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
) -> str:
    """导出 PyTorch 图像分类模型为 ONNX"""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError:
        raise ImportError("需要安装 torch 和 torchvision")

    checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    arch = checkpoint["arch"]
    num_classes = checkpoint["num_classes"]
    img_size = checkpoint["img_size"]
    classes = checkpoint["classes"]

    model_fn = getattr(models, arch)
    model = model_fn(weights=None)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if output_path is None:
        output_path = os.path.join(model_dir, "model.onnx")

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset_version,
        do_constant_folding=True,
    )

    meta_path = os.path.join(model_dir, "onnx_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "classes": classes, "img_size": img_size,
            "arch": arch, "model_type": "image_classification",
        }, f, indent=2, ensure_ascii=False)

    return output_path


def export_audio_model_to_onnx(
    model_dir: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
) -> str:
    """导出 PyTorch 音频分类模型为 ONNX"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("需要安装 torch")

    checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]
    in_channels = checkpoint["in_channels"]

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if output_path is None:
        output_path = os.path.join(model_dir, "model.onnx")

    dummy_input = torch.randn(1, in_channels, 64, 94)

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["audio_features"],
        output_names=["logits"],
        dynamic_axes={
            "audio_features": {0: "batch_size", 2: "freq", 3: "time"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    meta_path = os.path.join(model_dir, "onnx_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "classes": classes, "model_type": "audio_classification",
            "feature_type": checkpoint.get("feature_type", "mel_spectrogram"),
        }, f, indent=2, ensure_ascii=False)

    return output_path


def auto_export(model_path: str) -> str:
    """自动检测模型类型并导出 ONNX"""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        model_pt = os.path.join(model_path, "model.pt")

        if os.path.exists(model_pt):
            with open(config_path) as f:
                config = json.load(f)
            if "arch" in config:
                return export_image_model_to_onnx(model_path)
            else:
                return export_audio_model_to_onnx(model_path)
        else:
            return export_text_model_to_onnx(model_path)

    elif model_path.endswith(".pkl"):
        return export_tabular_model_to_onnx(model_path)

    else:
        raise ValueError(f"无法识别模型类型：{model_path}")
