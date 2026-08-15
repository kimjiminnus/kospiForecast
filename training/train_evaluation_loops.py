from src.visualisation import plot_losses
import torch
import torch.nn as nn


def train_loop(model, train_loader, criterion, optimiser, device):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimiser.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        train_loss += loss.item()
    return train_loss/len(train_loader)


def eval_loop(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
    return val_loss/len(val_loader)


def train_and_evaluate(epochs, model, train_loader, val_loader, criterion, optimiser, device, plot_loss=True):
    ave_train_losses = []
    ave_val_losses = []

    for epoch in range(epochs):
        train_loss = train_loop(model, train_loader, criterion, optimiser, device)
        ave_train_losses.append(train_loss)

        val_loss = eval_loop(model, val_loader, criterion, device)
        ave_val_losses.append(val_loss)

    if plot_loss:
        plot_losses(ave_train_losses, ave_val_losses)

    return model, ave_train_losses, ave_val_losses

