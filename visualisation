def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))

    if train_losses:
        plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")

    plt.yscale("log")

    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.title("Training vs Validation Loss")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()
