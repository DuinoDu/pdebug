from pdebug.otn import manager as otn_manager
from pdebug.utils.env import TORCH_INSTALLED, TORCHVISION_INSTALLED

import typer

if TORCH_INSTALLED:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, random_split

if TORCHVISION_INSTALLED:
    from torchvision import transforms
    from torchvision.datasets import MNIST

try:
    import pytorch_lightning
    from pytorch_lightning import (
        LightningDataModule,
        LightningModule,
        Trainer,
        seed_everything,
    )
except ModuleNotFoundError:
    pytorch_lightning = None
    LightningDataModule = object
    LightningModule = object
    Trainer = None


# Config #


# Data ##


class DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


# Model #


class Model(LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )


# Train #


@otn_manager.NODE.register(name="mnist_train")
def main(
    append_data: str = None,
    output: str = "xxx.onnx",
):
    """Train mnist in one file."""
    if pytorch_lightning is None:
        raise RuntimeError("pytorch-lightning is not installed.")
    seed_everything(1234)

    data = DataModule()
    model = Model()

    trainer = Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, datamodule=data)
    trainer.test(model, datamodule=data)

    return output


if __name__ == "__main__":
    typer.run(main)
