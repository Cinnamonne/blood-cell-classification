import argparse
import random
from pathlib import Path

import requests
from PIL import Image
from medmnist import BloodMNIST
from torchvision import transforms

from show_bloodmnist import show_bloodmnist_images_by_indices


API_URL = "http://localhost:3000/predict"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

MAX_IMAGES_TO_DISPLAY = 20


# Transformacje takie same jak podczas treningu modelu
# Resize dopasowuje obraz do rozmiaru BloodMNIST
# ToTensor zamienia obraz na tensor PyTorch
# Normalize skaluje wartości pikseli do zakresu oczekiwanego przez model
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


def send_image(image_tensor, source_name, true_label=None):
    # Wysłanie obrazu do API BentoML
    # image_tensor.tolist() zamienia tensor na listę zgodną z formatem JSON
    response = requests.post(
        API_URL,
        json={"image": image_tensor.tolist()},
    )

    response.raise_for_status()
    result = response.json()

    print("Source:", source_name)

    if true_label is not None:
        print("True label:", true_label)

    print("Response:")
    print(result)
    print("-" * 50)

    return result


def load_test_dataset():
    # Wczytanie testowej części zbioru BloodMNIST
    return BloodMNIST(
        split="test",
        root="data",
        transform=transform,
        download=True,
    )


def run_dataset_examples_by_indices(indices, show_images=False):
    # Wczytanie datasetu z transformacjami potrzebnymi do predykcji
    test_dataset = load_test_dataset()

    predictions = []

    # Wysłanie każdego obrazu do API
    for index in indices:
        image, label = test_dataset[index]

        result = send_image(
            image_tensor=image,
            source_name=f"BloodMNIST test_dataset[{index}]",
            true_label=int(label[0]),
        )

        predictions.append(result)

    # Wyświetlenie obrazów tylko wtedy, gdy podano przełącznik --show-images
    if show_images:
        if len(indices) > MAX_IMAGES_TO_DISPLAY:
            raise ValueError(
                f"Można wyświetlić maksymalnie {MAX_IMAGES_TO_DISPLAY} obrazów naraz"
            )

        show_bloodmnist_images_by_indices(
            indices=indices,
            predictions=predictions,
        )


def run_dataset_first_example(show_images=False):
    # Pobranie pierwszego przykładu ze zbioru testowego
    run_dataset_examples_by_indices(
        indices=[0],
        show_images=show_images,
    )


def run_dataset_random_examples(number_of_examples, show_images=False):
    # Pobranie kilku losowych przykładów ze zbioru testowego
    test_dataset = load_test_dataset()

    if number_of_examples > len(test_dataset):
        raise ValueError("Liczba przykładów jest większa niż rozmiar datasetu")

    indices = random.sample(
        range(len(test_dataset)),
        k=number_of_examples,
    )

    run_dataset_examples_by_indices(
        indices=indices,
        show_images=show_images,
    )


def run_file_example(image_path):
    # Wczytanie obrazu z pliku i zamiana na RGB
    image = Image.open(image_path).convert("RGB")

    # Przygotowanie obrazu tak jak podczas treningu
    image_tensor = transform(image)

    send_image(
        image_tensor=image_tensor,
        source_name=str(image_path),
    )


def run_folder_example(folder_path):
    # Pobranie wszystkich obrazów z folderu
    image_paths = sorted(
        file_path
        for file_path in folder_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) == 0:
        print("Folder nie zawiera obsługiwanych plików obrazów")
        return

    for image_path in image_paths:
        run_file_example(image_path)


parser = argparse.ArgumentParser()

parser.add_argument(
    "input_value",
    nargs="?",
    default=None,
    help="Liczba przykładów z datasetu, ścieżka do obrazu albo ścieżka do folderu",
)

parser.add_argument(
    "--indices",
    nargs="+",
    type=int,
    default=None,
    help="Indeksy przykładów z BloodMNIST",
)

parser.add_argument(
    "--show-images",
    action="store_true",
    help="Wyświetla obrazy z BloodMNIST po wykonaniu predykcji",
)

args = parser.parse_args()


if args.indices is not None:
    run_dataset_examples_by_indices(
        indices=args.indices,
        show_images=args.show_images,
    )

elif args.input_value is None:
    run_dataset_first_example(
        show_images=args.show_images,
    )

elif args.input_value.isdigit():
    number_of_examples = int(args.input_value)

    if number_of_examples <= 0:
        raise ValueError("Liczba przykładów musi być większa od 0")

    run_dataset_random_examples(
        number_of_examples=number_of_examples,
        show_images=args.show_images,
    )

else:
    input_path = Path(args.input_value)

    if input_path.is_file():
        run_file_example(input_path)
    elif input_path.is_dir():
        run_folder_example(input_path)
    else:
        raise FileNotFoundError(f"Nie znaleziono ścieżki: {input_path}")