import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from Alexnet import Custom_AlexNet
from DenseNet import Custom_DenseNet
from EfficientNet import create_efficientnet_b0
from GoogLeNet import GoogLeNet
from inception import Custom_Inception_Net
from MobileNetV3 import create_mobilenetv3_large
from mobilenet import Custom_MobileNet_V2
from NASNet import create_nasnet_mobile
from RepVGG import create_repvgg
from resnet50 import ResNet50
from ResNext import ResNext
from SqueezeNet import create_squeezenet
from vgg19 import Custom_VGG
from Xception import Custom_xception


class CNNTester:
    def __init__(self, device="auto", num_classes=1000):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.num_classes = num_classes
        self.input_size = 224

        self.registry = {
            "alexnet": lambda: Custom_AlexNet(num_classes=self.num_classes),
            "vgg19": lambda: Custom_VGG(num_classes=self.num_classes),
            "resnet50": lambda: ResNet50(num_classes=self.num_classes),
            "inception": lambda: Custom_Inception_Net(num_classes=self.num_classes),
            "mobilenetv2": lambda: Custom_MobileNet_V2(num_classes=self.num_classes),
            "xception": lambda: Custom_xception(num_classes=self.num_classes),
            "resnext": lambda: ResNext(num_classes=self.num_classes),
            "repvgg": lambda: create_repvgg(num_classes=self.num_classes, deploy=False),
            "googlenet": lambda: GoogLeNet(num_classes=self.num_classes),
            "efficientnet": lambda: create_efficientnet_b0(num_classes=self.num_classes),
            "squeezenet": lambda: create_squeezenet(num_classes=self.num_classes),
            "densenet": lambda: Custom_DenseNet(num_classes=self.num_classes),
            "nasnet": lambda: create_nasnet_mobile(num_classes=self.num_classes),
            "mobilenetv3": lambda: create_mobilenetv3_large(num_classes=self.num_classes),
        }

        self.model_order = [
            "alexnet",
            "vgg19",
            "resnet50",
            "inception",
            "mobilenetv2",
            "xception",
            "resnext",
            "repvgg",
            "googlenet",
            "efficientnet",
            "squeezenet",
            "densenet",
            "nasnet",
            "mobilenetv3",
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_model(self, model_name):
        name = model_name.lower()
        if name not in self.registry:
            raise ValueError(f"Unknown model: {model_name}")
        model = self.registry[name]().to(self.device)
        model.eval()
        return model

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def _topk(self, logits, top_k):
        probs = F.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=top_k, dim=1)
        results = []
        for score, index in zip(values[0].tolist(), indices[0].tolist()):
            results.append(
                {
                    "class_id": index,
                    "label": f"class_{index}",
                    "score": score,
                }
            )
        return results

    def classify(self, model_name, image_path, top_k=5):
        model = self.get_model(model_name)
        x = self._load_image(image_path)
        top_k = max(1, min(top_k, self.num_classes))

        with torch.no_grad():
            start = time.perf_counter()
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        results = self._topk(logits, top_k)
        return {
            "model": model_name.lower(),
            "time_ms": elapsed_ms,
            "topk": results,
        }

    def compare_architectures(self, image_path, top_k=5):
        rows = []
        for model_name in self.model_order:
            try:
                result = self.classify(model_name=model_name, image_path=image_path, top_k=top_k)
                top1 = result["topk"][0]
                rows.append(
                    {
                        "model": model_name,
                        "class_id": top1["class_id"],
                        "score": top1["score"],
                        "time_ms": result["time_ms"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "model": model_name,
                        "class_id": "ERROR",
                        "score": 0.0,
                        "time_ms": -1.0,
                        "error": str(e),
                    }
                )
        return rows

    def benchmark(self, image_path, iterations=10, warmup=3):
        x = self._load_image(image_path)
        records = []
        for model_name in self.model_order:
            try:
                model = self.get_model(model_name)
                params = sum(p.numel() for p in model.parameters())

                with torch.no_grad():
                    for _ in range(warmup):
                        _ = model(x)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(iterations):
                        _ = model(x)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    total = time.perf_counter() - start

                avg_ms = (total * 1000.0) / iterations
                fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                records.append(
                    {
                        "model": model_name,
                        "params_m": params / 1_000_000.0,
                        "avg_ms": avg_ms,
                        "fps": fps,
                    }
                )
            except Exception as e:
                records.append(
                    {
                        "model": model_name,
                        "params_m": -1.0,
                        "avg_ms": -1.0,
                        "fps": 0.0,
                        "error": str(e),
                    }
                )
        return records

    @staticmethod
    def print_single(result):
        print(f"Model: {result['model']}")
        print(f"Inference time: {result['time_ms']:.2f} ms")
        print("-" * 72)
        for i, item in enumerate(result["topk"], start=1):
            print(f"{i:>2}. {item['label']:<16} (id={item['class_id']:<4})  {item['score'] * 100:6.2f}%")
        print("-" * 72)

    @staticmethod
    def print_compare(rows):
        print("CNN ARCHITECTURE COMPARISON")
        print("=" * 88)
        print(f"{'Model':<14}{'Top1 Class':<12}{'Top1 Prob(%)':<14}{'Time(ms)':<12}")
        print("-" * 88)
        for row in rows:
            if "error" in row:
                print(f"{row['model']:<14}{'ERROR':<12}{'-':<14}{'-':<12}")
            else:
                print(f"{row['model']:<14}{row['class_id']:<12}{row['score'] * 100:<14.2f}{row['time_ms']:<12.2f}")
        print("=" * 88)

    @staticmethod
    def print_benchmark(rows, iterations):
        print("ARCHITECTURE BENCHMARK")
        print("=" * 88)
        print(f"Iterations: {iterations}")
        print(f"{'Model':<14}{'Params(M)':<12}{'Avg Time(ms)':<14}{'Throughput(fps)':<16}")
        print("-" * 88)
        for row in rows:
            if "error" in row:
                print(f"{row['model']:<14}{'ERROR':<12}{'-':<14}{'-':<16}")
            else:
                print(f"{row['model']:<14}{row['params_m']:<12.2f}{row['avg_ms']:<14.2f}{row['fps']:<16.2f}")
        print("=" * 88)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified CNN model tester for 14 architectures")
    parser.add_argument("--model", "-m", type=str, default="efficientnet", help="single model name")
    parser.add_argument("--image", "-i", type=str, required=True, help="image path")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="number of top predictions")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="run device")
    parser.add_argument("--compare", "-c", action="store_true", help="run all 14 models and compare")
    parser.add_argument("--benchmark", "-b", action="store_true", help="run benchmark for all 14 models")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="benchmark iterations")
    parser.add_argument("--num_classes", type=int, default=1000, help="output classes for all models")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    tester = CNNTester(device=args.device, num_classes=args.num_classes)

    if args.compare:
        compare_rows = tester.compare_architectures(image_path=args.image, top_k=args.top_k)
        tester.print_compare(compare_rows)
        return

    if args.benchmark:
        benchmark_rows = tester.benchmark(image_path=args.image, iterations=args.iterations)
        tester.print_benchmark(benchmark_rows, iterations=args.iterations)
        return

    result = tester.classify(model_name=args.model, image_path=args.image, top_k=args.top_k)
    tester.print_single(result)


if __name__ == "__main__":
    main()
