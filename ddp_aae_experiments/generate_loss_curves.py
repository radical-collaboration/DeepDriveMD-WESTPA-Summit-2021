import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    paths = [
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/1-gpu_32-gbs",
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/2-gpu_64-gbs",
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/1-node_128-gbs",
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/2-node_256-gbs",
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/4-node_512-gbs",
        "/scratch/06079/tg853783/ddmd/runs/ddp_aae_experiments/8-node_1024-gbs",
    ]

    csv_files = [Path(p) / "loss.json" for p in paths]

    # Epoch to start plotting from so the end behavior is more clearly shown
    start_epoch = 10

    # For testing
    # csv_files = [Path("./test.json").resolve()]

    train_loss_eg_averages, train_loss_d_averages, valid_losses = [], [], []
    for csv_file in csv_files:
        with open(csv_file) as f:
            data = json.load(f)
        train_loss_eg_averages.append(data["train_loss_eg_average"])
        train_loss_d_averages.append(data["train_loss_d_average"])
        valid_losses.append(data["valid_loss"])

    titles = [
        "Reconstruction Training Loss",
        "Discriminator Training Loss",
        "Validation Loss",
    ]
    loss_types = [train_loss_eg_averages, train_loss_d_averages, valid_losses]

    for loss_type, title in zip(loss_types, titles):
        df = pd.DataFrame(
            {p.parent.name: loss for p, loss in zip(csv_files, loss_type)}
        )
        df[start_epoch:].plot(legend=True)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f'./img/{title.replace(" ","")}.png')
