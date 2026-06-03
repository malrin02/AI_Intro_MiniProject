import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.optimizer import Adam
from dataset.fashion_mnist import load_fashion_mnist
from simple_convnet import SimpleConvNet


def set_seed(seed=42):
    np.random.seed(seed)


def save_graph(train_values, test_values, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_values) + 1)
    plt.plot(epochs, train_values, marker="o", label="Train")
    plt.plot(epochs, test_values, marker="s", label="Test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "data")
    log_path = os.path.join(base_dir, "logs.txt")
    acc_graph_path = os.path.join(base_dir, "accuracy_graph.png")
    loss_graph_path = os.path.join(base_dir, "loss_graph.png")
    params_path = os.path.join(base_dir, "scratch_params.pkl")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Fashion MNIST Scratch CNN Training Log\n")
        f.write("=" * 50 + "\n")

    (x_train, t_train), (x_test, t_test) = load_fashion_mnist(
        data_dir=data_dir,
        normalize=True,
        flatten=False,
        one_hot_label=False,
    )

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    print("x_train:", x_train.shape)
    print("t_train:", t_train.shape)
    print("x_test:", x_test.shape)
    print("t_test:", t_test.shape)

    network = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={
            "filter_nums": (32, 32, 64, 64),
            "filter_size": 3,
            "pad": 1,
            "stride": 1,
        },
        hidden_size=256,
        output_size=10,
        weight_init_std="he",
        dropout_ratio=0.35,
        weight_decay_lambda=1e-4,
    )

    optimizer = Adam(lr=0.001)
    max_epochs = 30
    train_size = x_train.shape[0]
    batch_size = 128
    iter_per_epoch = max(train_size // batch_size, 1)
    max_iter = max_epochs * iter_per_epoch

    train_loss_list = []
    epoch_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in range(1, max_iter + 1):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        loss = network.loss(x_batch, t_batch, train_flg=False)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            epoch = i // iter_per_epoch
            if epoch in (18, 25):
                optimizer.lr *= 0.3

            train_acc = network.accuracy(x_train, t_train, batch_size=200)
            test_acc = network.accuracy(x_test, t_test, batch_size=200)
            epoch_loss = float(np.mean(train_loss_list[-iter_per_epoch:]))

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            epoch_loss_list.append(epoch_loss)

            log = (
                f"Epoch [{epoch:02d}/{max_epochs}] "
                f"Iter [{i}/{max_iter}] "
                f"LR: {optimizer.lr:.6f}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Train Acc: {train_acc * 100:.2f}%, "
                f"Test Acc: {test_acc * 100:.2f}%"
            )
            print(log)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log + "\n")

    print("Calculating final accuracy...")
    final_train_acc = network.accuracy(x_train, t_train, batch_size=200)
    final_test_acc = network.accuracy(x_test, t_test, batch_size=200)

    final_log = (
        "\nFinal Result\n"
        f"Train Accuracy: {final_train_acc * 100:.2f}%\n"
        f"Test Accuracy: {final_test_acc * 100:.2f}%\n"
    )
    print(final_log)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(final_log)

    save_graph(
        [acc * 100 for acc in train_acc_list],
        [acc * 100 for acc in test_acc_list],
        "Fashion MNIST Scratch CNN Accuracy",
        "Accuracy (%)",
        acc_graph_path,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(epoch_loss_list) + 1),
        epoch_loss_list,
        marker="o",
        label="Train Loss",
    )
    plt.title("Fashion MNIST Scratch CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_graph_path)
    plt.close()

    network.save_params(params_path)

    print("Saved files:")
    print(" - logs.txt")
    print(" - accuracy_graph.png")
    print(" - loss_graph.png")
    print(" - scratch_params.pkl")


if __name__ == "__main__":
    main()
