import math

import matplotlib.pyplot as plt
from medmnist import BloodMNIST


CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


def load_raw_bloodmnist_dataset(split="test", root="data"):
    # Wczytanie datasetu bez transformacji, żeby wyświetlać oryginalne obrazy
    return BloodMNIST(
        split=split,
        root=root,
        download=True,
    )


def show_bloodmnist_images_by_indices(indices, predictions=None, split="test", root="data"):
    # Wyświetlenie obrazów z BloodMNIST dla podanych indeksów
    dataset = load_raw_bloodmnist_dataset(split=split, root=root)

    number_of_images = len(indices)
    columns = min(4, number_of_images)
    rows = math.ceil(number_of_images / columns)

    plt.figure(figsize=(2 * columns, 2 * rows))

    for plot_index, dataset_index in enumerate(indices, start=1):
        image, label = dataset[dataset_index]

        true_label_id = int(label[0])
        true_label_name = CLASS_NAMES[true_label_id]

        plt.subplot(rows, columns, plot_index)
        plt.imshow(image)
        plt.axis("off")

        if predictions is None:
            title = (
                f"index={dataset_index}\n"
                f"true={true_label_name}"
            )
        else:
            prediction = predictions[plot_index - 1]

            predicted_label_id = prediction["predicted_class_id"]
            predicted_label_name = prediction["predicted_class_name"]

            title = (
                f"index={dataset_index}\n"
                f"true={true_label_name}\n"
                f"pred={predicted_label_name}"
            )

        plt.title(title)

    plt.tight_layout()
    plt.show()