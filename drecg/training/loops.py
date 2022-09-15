
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

class SanityCallback(pl.Callback):
    def on_sanity_check_start(self, _trainer, pl_module) -> None:
        pl_module.on_sanity = True

    def on_sanity_check_end(self, _trainer, pl_module) -> None:
        pl_module.on_sanity = False

def train_lightning(lit_model, dataloader, val_dataloader, test_dl, epochs=10):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', logger=tb_logger, callbacks=[SanityCallback()])
    trainer.fit(lit_model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    trainer.test(dataloaders=test_dl)
    return trainer

def train_full(model, dataloader, device, epochs=10):
    dl_len = len(dataloader)

    model.to(device)
    optimizer = Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(range(epochs*dl_len))

    hist_loss = []
    hist_acc = []

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for (img_a_batch, img_b_batch), label_batch in dataloader:
            img_a_batch = img_a_batch.to(device)
            img_b_batch = img_b_batch.to(device)
            label_batch = label_batch.to(device).reshape(-1, 1).to(torch.float32)

            optimizer.zero_grad()
            y_pred = model((img_a_batch, img_b_batch))
            loss = loss_fn(y_pred, label_batch)
            loss.backward()
            optimizer.step()
            
            y_preds = torch.round(torch.sigmoid(y_pred))
            acc = (y_preds == label_batch).sum().item() / len(label_batch)
            epoch_acc += acc
            epoch_loss += loss.item()

            progress_bar.update(1)
        hist_loss.append(epoch_loss / dl_len)
        hist_acc.append(epoch_acc / dl_len)
        print(f'Epoch {epoch}: Loss {epoch_loss / dl_len}, Acc {epoch_acc / dl_len}')

    return hist_loss, hist_acc

def train(model, dataloader_same, dataloader_diff, device, epochs=10):
    same_len = len(dataloader_same)
    diff_len = len(dataloader_diff)

    model.to(device)
    optimizer = Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(range(epochs*same_len))

    
    if same_len != diff_len:
        raise ValueError('Dataloaders should have the same length')

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        iter_same = iter(dataloader_same)
        iter_diff = iter(dataloader_diff)
        epoch_loss = 0
        model.train()
        for i in range(same_len):
            batch_same = next(iter_same)
            batch_diff = next(iter_diff)

            same_pairs = batch_same[0][0].to(device), batch_same[0][1].to(device)
            diff_pairs = batch_diff[0][0].to(device), batch_diff[0][1].to(device)

            optimizer.zero_grad()
            y_same = model(same_pairs)
            y_diff = model(diff_pairs)

            loss_same = loss_fn(y_same, batch_same[1].to(device).reshape(-1, 1).to(torch.float32))
            loss_diff = loss_fn(y_diff, batch_diff[1].to(device).reshape(-1, 1).to(torch.float32))
            loss = loss_same + loss_diff
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            progress_bar.update(1)

        print(f'Epoch {epoch}: {epoch_loss / same_len}')


