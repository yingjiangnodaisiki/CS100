import os
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 图像分类推理脚本")
    parser.add_argument("--image", type=str, required=True, help="待预测图片路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="训练好的模型权重文件")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    return parser.parse_args()


def load_model(backbone: str, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet34":
        model = models.resnet34(weights=None)
    else:
        model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, class_names


def preprocess_image(image_path: str, img_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(image_path: str, checkpoint_path: str, backbone: str = "resnet18", img_size: int = 224):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(backbone, checkpoint_path, device)
    input_tensor = preprocess_image(image_path, img_size=img_size).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = class_names[pred_idx.item()]
    confidence = conf.item()
    print(f"图片: {image_path}")
    print(f"预测类别: {pred_class}")
    print(f"置信度: {confidence:.4f}")
    return pred_class, confidence


def main():
    args = parse_args()
    predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()

