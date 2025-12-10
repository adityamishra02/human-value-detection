import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from torch import nn
from sklearn.metrics import f1_score
import time
from transformers import DebertaModel


class AdamSmithValues(pl.LightningModule):
    def __init__(self, model_name='microsoft/deberta-v3-base', lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = DebertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        num_labels_sub1 = 19   
        num_labels_sub2 = 38  

        self.head_sub1 = nn.Linear(self.backbone.config.hidden_size, num_labels_sub1)
        self.head_sub2 = nn.Linear(self.backbone.config.hidden_size, num_labels_sub2)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self._last_time = None 

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        ids, mask, y1, y2 = batch
        outputs = self.backbone(ids, attention_mask=mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits1 = self.head_sub1(pooled).squeeze(-1)
        logits2 = self.head_sub2(pooled).squeeze(-1)

        loss1 = self.criterion(logits1, y1.float())
        loss2 = self.criterion(logits2, y2.float())
        loss = (loss1 + loss2) / 2


        if self._last_time is not None:
            print(f"⏱ Batch {batch_idx}: {time.time() - self._last_time:.3f} sec")
        self._last_time = time.time()

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def forward(self, ids, mask):
        outputs = self.backbone(input_ids=ids, attention_mask=mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head_sub1(pooled), self.head_sub2(pooled)

    def _step(self, batch):
        if isinstance(batch, dict):
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            y1 = batch["labels_sub1"]
            y2 = batch["labels_sub2"]
        else:
            ids, mask, y1, y2 = batch[:4]

        o1, o2 = self(ids, mask)
        loss1 = self.criterion(o1, y1)
        loss2 = self.criterion(o2, y2)
        return loss1 + loss2, o1, o2, y1, y2

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, o1, o2, y1, y2 = self._step(batch)
        p1 = (torch.sigmoid(o1) > 0.5).float()
        p2 = (torch.sigmoid(o2) > 0.5).float()

        return {
            "val_loss": loss.detach(),
            "p1": p1.cpu(),
            "p2": p2.cpu(),
            "y1": y1.cpu(),
            "y2": y2.cpu(),
        }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        y1 = torch.cat([x["y1"] for x in outputs]).numpy()
        p1 = torch.cat([x["p1"] for x in outputs]).numpy()
        y2 = torch.cat([x["y2"] for x in outputs]).numpy()
        p2 = torch.cat([x["p2"] for x in outputs]).numpy()

        f1_1 = f1_score(y1, p1, average="macro")
        f1_2 = f1_score(y2, p2, average="macro")

        self.log("val_loss", loss.float(), prog_bar=True)
        self.log("val_sub1_f1", float(f1_1), prog_bar=True)
        self.log("val_sub2_f1", float(f1_2), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
