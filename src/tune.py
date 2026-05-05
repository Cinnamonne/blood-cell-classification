import optuna
import torch
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

from datamodule import BloodMNISTDataModule
from model import BloodCellCNN


def train_one_model(
    learning_rate: float,
    weight_decay: float,
    trial_name: str,
    max_epochs: int = 50,
) -> float:
    """
    Trenuje jeden model dla:
    - learning_rate,
    - weight_decay.

    Zwraca najlepszy val_loss osiągnięty podczas treningu.
    Optuna będzie minimalizować tę wartość.
    """

    L.seed_everything(42)

    data_module = BloodMNISTDataModule(
        data_dir="data",
        batch_size=128,
        num_workers=0,
    )

    model = BloodCellCNN(
        num_classes=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    wandb_logger = WandbLogger(
        project="mlops-bloodmnist",
        name=trial_name,
        group="optuna-lr-weight-decay-50epochs",
        log_model=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[early_stopping],
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    # Bierzemy najlepszy val_loss z całego treningu.
    best_val_loss = early_stopping.best_score.item()

    wandb.finish()

    return best_val_loss


def objective(trial: optuna.Trial) -> float:
    """
    Funkcja celu Optuny.

    Optuna wybiera learning_rate i weight_decay,
    trenuje model, a potem zwraca najlepszy val_loss.
    """

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True,
    )

    trial_name = (
        f"optuna-lr-wd-50epochs-trial-{trial.number}"
        f"-lr-{learning_rate:.6f}"
        f"-wd-{weight_decay:.6f}"
    )

    best_val_loss = train_one_model(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        trial_name=trial_name,
        max_epochs=50,
    )

    return best_val_loss


def main():
    """
    Optymalizacja learning_rate i weight_decay
    dla treningu z max_epochs = 50 i EarlyStopping.
    """

    torch.set_float32_matmul_precision("medium")

    # Baseline, czyli punkt odniesienia.
    baseline_lr = 1e-3
    baseline_weight_decay = 1e-4

    print("Trening baseline dla 50 epok...")
    baseline_val_loss = train_one_model(
        learning_rate=baseline_lr,
        weight_decay=baseline_weight_decay,
        trial_name="baseline-lr-0.001-wd-0.0001-50epochs",
        max_epochs=50,
    )

    # Tworzymy badanie
    # direction='minimize' - chcemy jak najmniejszy val_loss
    study = optuna.create_study(direction="minimize")

    #ustawiono 10 pr
    study.optimize(
        objective,
        n_trials=10,
    )

    results = study.trials_dataframe()
    results.to_csv("optuna_results_lr_weight_decay_50epochs.csv", index=False)

    print("\nBaseline dla 50 epok:")
    print(f"learning_rate: {baseline_lr}")
    print(f"weight_decay: {baseline_weight_decay}")
    print(f"best val_loss: {baseline_val_loss:.4f}")

    print("\nNajlepszy wynik Optuny:")
    print(f"learning_rate: {study.best_params['learning_rate']}")
    print(f"weight_decay: {study.best_params['weight_decay']}")
    print(f"best val_loss: {study.best_value:.4f}")

    improvement = baseline_val_loss - study.best_value

    print("\nRóżnica względem baseline:")
    print(f"poprawa val_loss: {improvement:.4f}")

    print("\nWyniki zapisano w pliku:")
    print("optuna_results_lr_weight_decay_50epochs.csv")


if __name__ == "__main__":
    main()