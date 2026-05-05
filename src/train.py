import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from datamodule import BloodMNISTDataModule
from model import BloodCellCNN


def main():
    """
    Główna funkcja treningu:

    - tworzy DataModule,
    - tworzy model,
    - ustawia logowanie do WandB,
    - uruchamia trening,
    - testuje najlepszy zapisany model.
    """

    L.seed_everything(42) # Ustawiamy seed dla powtarzalności wyników

    # DataModule - odpowiada za dane i DataLoadery
    data_module = BloodMNISTDataModule(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    # Tworzymy model CNN dla 8 klas BloodMNIST
    model = BloodCellCNN(
        num_classes=8,
        learning_rate=0.0015435312235389442,
        weight_decay=1.053809851988176e-05,
    )

    # WandB zapisuje wyniki treningu w panelu
    wandb_logger = WandbLogger(
        project="mlops-bloodmnist",
        name="cnn-adamw-optimized-lr-wd-50epochs",
        log_model=False,
    )

    # Zapis najlepszego modelu według najmniejszego val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
    )

    # Zatrzymuje trening, jeśli val_loss długo się nie poprawia
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    )

    # Loguje learning rate do WandB
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
    )

    # Trainer steruje całym treningiem
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
        ],
        log_every_n_steps=10,
    )

    # Uruchomienie treningu
    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Testujemy najlepszy zapisany model
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path="best",
    )

    print("Najlepszy model zapisany tutaj:")
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()