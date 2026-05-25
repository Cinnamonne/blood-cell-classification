from pathlib import Path
import sys
import time
import copy

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model import BloodCellCNN


CHECKPOINT_PATH = "checkpoints/best.ckpt"


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


test_dataset = BloodMNIST(
    split="test",
    root="data",
    transform=transform,
    target_transform=lambda y: int(y[0]),
    download=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
)


def load_model():
    model = BloodCellCNN.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.eval()

    return model


def evaluate(model):
    predictions = []
    labels_all = []

    start = time.perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.tolist())
            labels_all.extend(labels.tolist())

    end = time.perf_counter()

    f1 = f1_score(
        labels_all,
        predictions,
        average="macro",
    )

    return f1, end - start


def model_size_mb(model, path):
    torch.save(model.state_dict(), path)
    size = Path(path).stat().st_size / (1024 * 1024)
    Path(path).unlink()

    return size


def count_nonzero_params(model):
    nonzero = 0
    total = 0

    for param in model.parameters():
        nonzero += torch.count_nonzero(param).item()
        total += param.numel()

    return nonzero, total


def apply_unstructured_pruning(model, amount=0.3):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(
                module,
                name="weight",
                amount=amount,
            )
            prune.remove(module, "weight")

    return model


def apply_structured_pruning(model, amount=0.3):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0,
            )
            prune.remove(module, "weight")

        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0,
            )
            prune.remove(module, "weight")

    return model


def print_quantization_row(name, f1, inference_time, size):
    print(f"{name:<10} | F1: {f1:.4f} | time: {inference_time:.4f} s | size: {size:.4f} MB")


def print_pruning_row(name, f1, inference_time, nonzero, total):
    print(
        f"{name:<25} | "
        f"F1: {f1:.4f} | "
        f"time: {inference_time:.4f} s | "
        f"non-zero params: {nonzero}/{total}"
    )


def main():
    float32_model = load_model()

    float32_f1, float32_time = evaluate(float32_model)
    float32_size = model_size_mb(float32_model, "float32_model.pt")

    #fp16 oznacza float16, czyli 16-bitowe liczby zmiennoprzecinkowe
    #na cpu model jest przy ewaluacji cofany do float32, bo cpu nie obsługuje fp16 tak wygodnie jak gpu
    fp16_model = copy.deepcopy(float32_model).half()
    fp16_f1, fp16_time = evaluate(fp16_model.float())
    fp16_size = model_size_mb(fp16_model, "fp16_model.pt")

    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(float32_model),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    int8_f1, int8_time = evaluate(int8_model)
    int8_size = model_size_mb(int8_model, "int8_model.pt")

    print("\nQuantization results")
    print("-" * 70)
    print_quantization_row("float32", float32_f1, float32_time, float32_size)
    print_quantization_row("fp16", fp16_f1, fp16_time, fp16_size)
    print_quantization_row("int8", int8_f1, int8_time, int8_size)

    baseline_nonzero, baseline_total = count_nonzero_params(float32_model)

    unstructured_model = apply_unstructured_pruning(
        copy.deepcopy(float32_model),
        amount=0.3,
    )
    unstructured_f1, unstructured_time = evaluate(unstructured_model)
    unstructured_nonzero, unstructured_total = count_nonzero_params(unstructured_model)

    structured_model = apply_structured_pruning(
        copy.deepcopy(float32_model),
        amount=0.3,
    )
    structured_f1, structured_time = evaluate(structured_model)
    structured_nonzero, structured_total = count_nonzero_params(structured_model)

    print("\nPruning results")
    print("-" * 90)
    print_pruning_row("baseline", float32_f1, float32_time, baseline_nonzero, baseline_total)
    print_pruning_row("unstructured pruning", unstructured_f1, unstructured_time, unstructured_nonzero, unstructured_total)
    print_pruning_row("structured pruning", structured_f1, structured_time, structured_nonzero, structured_total)


if __name__ == "__main__":
    main()