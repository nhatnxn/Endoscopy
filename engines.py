
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import dice_score

def train_fn(loaders, model, criterion, optimizer, lr_scheduler, start_epoch, total_epochs, device, device_ids, save_path):
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    print("Epochs: {}\n".format(total_epochs))
    best_epoch = 1
    best_dice = 0.0
    history = {
        "train": {"loss": [], "dice": []}, 
        "eval": {"loss": [], "dice": []}, 
        "lr": []
    }

    for epoch in range(start_epoch, total_epochs + 1):
        head = "epoch {:3}/{:3}".format(epoch, total_epochs)
        print(head + "\n" + "-"*(len(head)))

        model.train()
        running_loss = 0.0
        running_dice = 0.0
        for images, masks in tqdm.tqdm(loaders["train"]):
            images, masks = images.to(device), masks.to(device).squeeze(1).transpose(1, 3).transpose(2, 3)

            optimizer.zero_grad()

            outputs5, outputs4, outputs3 = model(images)
            outputs5, outputs4, outputs3 = outputs5, F.interpolate(outputs4, size=(masks.shape[2], masks.shape[3]), mode="bilinear"), F.interpolate(outputs3, size=(masks.shape[2], masks.shape[3]), mode="bilinear")
            preds5, preds4, preds3 = torch.sigmoid(outputs5), torch.sigmoid(outputs4), torch.sigmoid(outputs3)
            loss = criterion(preds5, masks) + criterion(preds4, masks) + criterion(preds3, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_score(masks.cpu().numpy(), preds5.cpu().detach().numpy())

        epoch_loss = running_loss/len(loaders["train"])
        epoch_dice = running_dice/len(loaders["train"])
        history["train"]["loss"].append(epoch_loss)
        history["train"]["dice"].append(epoch_dice)
        print("{:5} - loss: {:.6f} dice: {:.6f}".format("train", epoch_loss, epoch_dice))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_dice = 0.0
            for images, masks in tqdm.tqdm(loaders["eval"]):
                images, masks = images.to(device), masks.to(device).squeeze(1).transpose(1, 3).transpose(2, 3)

                outputs5, outputs4, outputs3 = model(images)
                outputs5, outputs4, outputs3 = outputs5, F.interpolate(outputs4, size=(masks.shape[2], masks.shape[3]), mode="bilinear"), F.interpolate(outputs3, size=(masks.shape[2], masks.shape[3]), mode="bilinear")
                preds5, preds4, preds3 = torch.sigmoid(outputs5), torch.sigmoid(outputs4), torch.sigmoid(outputs3)
                loss = criterion(preds5, masks) + criterion(preds4, masks) + criterion(preds3, masks)

                running_loss += loss.item()
                running_dice += dice_score(masks.cpu().numpy(), preds5.cpu().detach().numpy())

        epoch_loss = running_loss/len(loaders["eval"])
        epoch_dice = running_dice/len(loaders["eval"])
        history["eval"]["loss"].append(epoch_loss)
        history["eval"]["dice"].append(epoch_dice)
        print("{:5} - loss: {:.6f} dice: {:.6f}".format("eval", epoch_loss, epoch_dice))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        lr_scheduler.step(epoch_loss)

        if epoch_dice > best_dice:
            best_epoch = epoch
            best_dice = epoch_dice

            state_dicts = {
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(), 
                "lr_scheduler_state_dict": lr_scheduler.state_dict(), 
                "best_metric": best_dice
            }
            torch.save(state_dicts, save_path)

    with open("{}.json".format(save_path[:-3]), "w") as f:
        json.dump(history, f)
    print("\nFinish: - Best Epoch: {:3} - Best DICE: {:.6f}\n".format(best_epoch, best_dice))