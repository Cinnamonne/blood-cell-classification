from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import lightning as L
from medmnist import INFO, BloodMNIST
import os

class BloodMNISTDataModule(L.LightningDataModule):
    """
    DataModule odpowiada za dane:
    - pobranie datasetu,
    - przygotowanie transformacji,
    - utworzenie DataLoaderów dla treningu, walidacji i testu.
    """

    def __init__(
            self,
            data_dir: str = "data",
            batch_size: int = 128,
            num_workers: int = 0, #liczba dodatkowych procesów do ładowania danych
    ):
        super().__init__() #konstruktor klasy bazowej LightningDataModule

        self.data_dir = data_dir #folder do zapisu datasetu
        self.batch_size = batch_size #liczba obrazów w jednej paczce danych

        self.num_workers = num_workers
        self.info = INFO["bloodmnist"] #info o zbiorze BloodMNIST z biblioteki MedMNIST

        self.num_classes = len(self.info["label"]) #liczba klas - typów komórek krwi
        self.n_channels = self.info["n_channels"]# liczba kanałów obrazu, BloodMNIST ma obrazy RGB, więc są 3 kanały

        #zmienne na datasety, najpierw puste i zostaną utworzone w setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        #transformacje po kolei wykonywane na każdym obrazie
        self.transform = transforms.Compose(
            [
                #zamienia obraz na tensor PyTorch, kształt obrazu zmieni się na [kanały, wysokość, szerokość]
                transforms.ToTensor(),

                # Normalizuje wartości pikseli, obraz RGB ma 3 kanały, więc podajemy 3 średnie i 3 odchylenia
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def prepare_data(self):
        """
        Pobiera dane na dysk.
        Lightning uruchamia tę metodę tylko do pobierania danych,
        a nie do tworzenia obiektów datasetu używanych w treningu
        """
        os.makedirs(self.data_dir, exist_ok=True)
        BloodMNIST(
            split="train",
            root=self.data_dir,
            download=True,
        )

        BloodMNIST(
            split="val",
            root=self.data_dir,
            download=True,
        )

        BloodMNIST(
            split="test",
            root=self.data_dir,
            download=True,
        )

    def setup(self, stage: Optional[str] = None):
        """
        Tworzy datasety używane później przez DataLoadery.
        Mamy trzy części:
        - train: dane treningowe,
        - val: dane walidacyjne,
        - test: dane testowe.
        """

        self.train_dataset = BloodMNIST(
            split="train",
            root=self.data_dir,
            transform=self.transform,

            #MedMNIST zwraca etykietę jako małą tablicę, np. [3]
            #Model potrzebuje zwykłej liczby całkowitej, np. 3
            target_transform=lambda y: int(y[0]),
            download=False,
        )

        self.val_dataset = BloodMNIST(
            split="val",
            root=self.data_dir,
            transform=self.transform,
            target_transform=lambda y: int(y[0]),
            download=False,
        )

        self.test_dataset = BloodMNIST(
            split="test",
            root=self.data_dir,
            transform=self.transform,
            target_transform=lambda y: int(y[0]),
            download=False,
        )

    def train_dataloader(self):
        """
        Zwraca DataLoader dla treningu.
        shuffle=True miesza dane w każdej epoce.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,

            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """
        Zwraca DataLoader dla walidacji.
        shuffle=False, bo podczas walidacji nie trzeba mieszać danych.
        """

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        """
        Zwraca DataLoader dla testu.
        Test służy do końcowej oceny modelu po treningu.
        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    """
    Ten fragment uruchamia się, gdy odpalamy ten plik bezpośrednio:
    python src\\datamodule.py
    Służy do szybkiego sprawdzenia, czy dane działają.
    """

    dm = BloodMNISTDataModule()
    dm.prepare_data() # Pobranie danych
    dm.setup() # Utworzenie datasetów

    # Pobranie jednego batcha danych treningowych
    images, labels = next(iter(dm.train_dataloader()))

    # Sprawdzenie kształtów danych
    print("Shape obrazów:", images.shape)
    print("Shape etykiet:", labels.shape)

    # Sprawdzenie informacji o zbiorze
    print("Liczba klas:", dm.num_classes)
    print("Kanały obrazu:", dm.n_channels)
    print("Nazwy klas:", dm.info["label"])