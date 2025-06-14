"""
Benchmark script for NHSG117K GNN baselines.

Usage
-----
```bash
python src/gnl_transformer/training.py  \
    --root /path/to/NHSG117K \
    --epochs 100 \
    ...
```
"""

__all__ = [
    "NHSG117K_Lit",
    "LitGNN",
    "summarise_csv",
    "run_experiment",
]

import os, time
import numpy as np
import pandas as pd
from pathlib import Path
# from argparse import ArgumentParser
# from typing import List, Dict

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision

from gnl_transformer import NHSG117K

# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class NHSG117K_Lit(pl.LightningDataModule):
    def __init__(self, root, batch_size, seeds, subset_ratio = None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.seeds = seeds
        self.subset_ratio = subset_ratio

    def prepare_data(self):
        # trigger initial download / processing
        _ = NHSG117K(self.root)

    def setup(self, stage=None):
        self.splits = []
        full_dataset = NHSG117K(self.root)
        y_all_full = full_dataset._data.y.numpy()

        # If a subset_ratio is given, downsample the dataset before splitting
        if self.subset_ratio is not None:
            np.random.seed(42)  # for reproducible downsampling
            final_indices = []
            classes, counts = np.unique(y_all_full, return_counts=True)

            for cls, count in zip(classes, counts):
                class_indices = np.where(y_all_full == cls)[0]
                
                # If class size > 100, downsample it
                if count > 100:
                    num_to_sample = int(count * self.subset_ratio)
                    sampled_indices = np.random.choice(
                        class_indices, num_to_sample, replace=False
                    )
                    final_indices.extend(sampled_indices)
                else:
                    # Otherwise, keep all samples from that class
                    final_indices.extend(class_indices)
            
            # The working dataset is a subset of the full one
            self.dataset = Subset(full_dataset, sorted(final_indices))
            y_all = y_all_full[sorted(final_indices)]
        else:
            # Otherwise, use the full dataset
            self.dataset = full_dataset
            y_all = y_all_full

        self.y = torch.tensor(y_all)

        for seed in self.seeds:
            # 1) train/test
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=0.8, test_size=0.2, random_state=seed
            )
            # Generate indices relative to the (potentially downsampled) dataset
            idx_train, idx_tmp = next(splitter.split(np.zeros_like(y_all), y_all))
            y_tmp = y_all[idx_tmp]

            # 2) decide whether to stratify val/test
            #    if any class has fewer than 2 samples → fallback 
            counts = np.bincount(y_tmp)
            if counts.min() < 2:
                # plain ShuffleSplit 50/50
                ss = ShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
                rel_val, rel_test = next(ss.split(idx_tmp))
                idx_val = idx_tmp[rel_val]
                idx_test = idx_tmp[rel_test]
            else:
                # safe to Stratify
                splitter_val = StratifiedShuffleSplit(
                    n_splits=1, train_size=0.5, test_size=0.5, random_state=seed
                )
                rel_val, rel_test = next(splitter_val.split(
                    np.zeros_like(y_tmp), y_tmp
                ))
                idx_val = idx_tmp[rel_val]
                idx_test = idx_tmp[rel_test]

            self.splits.append((
                Subset(self.dataset, idx_train),
                Subset(self.dataset, idx_val),
                Subset(self.dataset, idx_test),
            ))

    def train_dataloader(self, seed_idx=0):
        return DataLoader(self.splits[seed_idx][0], batch_size=self.batch_size)

    def val_dataloader(self, seed_idx=0):
        return DataLoader(self.splits[seed_idx][1], batch_size=self.batch_size)

    def test_dataloader(self, seed_idx=0):
        return DataLoader(self.splits[seed_idx][2], batch_size=self.batch_size)

# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class LitGNN(pl.LightningModule):
    def __init__(self, model, train_hp):
        super().__init__()

        self.model = model
        self.save_hyperparameters({"model": self.model.__class__.__name__})
        self.save_hyperparameters(model.hps)
        self.save_hyperparameters(train_hp)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.topks = [1, 2]

        num_classes = model.hps['dim_out']
        self.train_acc = Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.val_accs = torch.nn.ModuleList([
            Accuracy(top_k=k, task="multiclass", num_classes=num_classes) for k in self.topks
        ])
        self.val_f1_macro = F1Score(average="macro", task="multiclass", num_classes=num_classes)
        self.val_f1_micro = F1Score(average="micro", task="multiclass", num_classes=num_classes)
        self.val_auc = AUROC(average="macro", task="multiclass", num_classes=num_classes)
        self.val_ap = AveragePrecision(average="macro", task="multiclass", num_classes=num_classes)

        self._train_samples = 0
        self._epoch_start_time = 0.0

    # optimiser + scheduler (Warm Restarts)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr_init, amsgrad=True, weight_decay=0.0
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.hparams.t0, T_mult=self.hparams.t_mult, eta_min=self.hparams.lr_min
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    # forward + shared step
    def forward(self, data):
        return self.model(data)

    def _step(self, batch, stage: str):
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        y = batch.y.view(-1)

        if stage == "train":
            self.train_acc.update(logits, y)
            self._train_samples += y.size(0)
        else:
            for acc in self.val_accs:
                acc.update(logits, y)
            self.val_f1_macro.update(logits, y)
            self.val_f1_micro.update(logits, y)
            self.val_auc.update(logits, y)
            self.val_ap.update(logits, y)

        self.log(f"{stage}_loss", loss, 
                 on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=batch.num_graphs)
        return loss

    # training / val / test hooks
    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    # epoch boundary hooks 
    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()
        self._train_samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self):
        # ---- core metrics
        self.log("train_top1", self.train_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        # ---- wall time, throughput, GPU memory
        wall_time = time.perf_counter() - self._epoch_start_time
        throughput = self._train_samples / wall_time if wall_time > 0 else 0.0
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            mem_gb = 0.0

        self.log("train_wall_time_s", wall_time, sync_dist=True)
        self.log("train_throughput_samples_s", throughput, sync_dist=True)
        self.log("train_gpu_mem_gb", mem_gb, sync_dist=True)

    def on_validation_epoch_end(self):
        for k, acc in zip(self.topks, self.val_accs):
            self.log(f"val_top{k}", acc.compute(), sync_dist=True); acc.reset()
        self.log("val_macro_f1",  self.val_f1_macro.compute(), sync_dist=True)
        self.log("val_micro_f1",  self.val_f1_micro.compute(), sync_dist=True)
        self.log("val_macro_auc", self.val_auc.compute(), sync_dist=True)
        self.log("val_macro_ap",  self.val_ap.compute(), sync_dist=True)
        self.val_f1_macro.reset(); self.val_f1_micro.reset()
        self.val_auc.reset(); self.val_ap.reset()

    def on_test_epoch_end(self):
        for k, acc in zip(self.topks, self.val_accs):
            self.log(f"test_top{k}", acc.compute(), sync_dist=True)
        self.log("test_macro_f1",  self.val_f1_macro.compute(), sync_dist=True)
        self.log("test_micro_f1",  self.val_f1_micro.compute(), sync_dist=True)
        self.log("test_macro_auc", self.val_auc.compute(), sync_dist=True)
        self.log("test_macro_ap",  self.val_ap.compute(), sync_dist=True)
    
    def on_fit_end(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.save_hyperparameters({
            "trainable_params": num_params,
        })

# ---------------------------------------------------------------------------
# Helper for summarising seeds → mean ± std CSV
# ---------------------------------------------------------------------------
def summarise_csv(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    rows = [{"metric": k, "mean": means[k], "std": stds[k]} for k in means.index]
    pd.DataFrame(rows).to_csv(csv_out, index=False)


# --- run experiment interactive wrapper --- #
def run_experiment(model, train_hp):
    # ---------- house-keeping ----------
    save_path = Path(train_hp['save_dir'])
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"⏳  Loading NHSG117K from {train_hp['root']}, saving to {save_path}, "
          f"batch={train_hp['batch_size']}, seeds={train_hp['seeds']}")

    # ---------- data ----------
    dm = NHSG117K_Lit(
        root=train_hp['root'],
        batch_size=train_hp['batch_size'],
        seeds=train_hp['seeds'],
        subset_ratio=train_hp.get('subset_ratio', None),
    )
    dm.prepare_data(); dm.setup()

    model_name = model.__class__.__name__.lower()
    print(f"\n🧠  ▶ Training {model_name} …")
    summaries = []

    for seed_idx, seed in enumerate(train_hp['seeds']):
        pl.seed_everything(seed, workers=True)

        train_loader = dm.train_dataloader(seed_idx)
        val_loader   = dm.val_dataloader(seed_idx)
        test_loader  = dm.test_dataloader(seed_idx)
        print(f"   • Seed {seed}: {dm.dataset.num_classes} classes, "
              f"{dm.dataset.num_node_features} input features.")


        lit = LitGNN(model, train_hp)

        logger = TensorBoardLogger(
            save_path / "tb_logs" / model_name,
            name=f"{model_name}_seed{seed}"
        )
        ckpt_dir = save_path / model_name / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_top1 = ModelCheckpoint(
            monitor="val_top1", mode="max", dirpath=ckpt_dir,
            filename="best-top1-{epoch:03d}-{val_top1:.4f}"
        )
        ckpt_f1 = ModelCheckpoint(
            monitor="val_macro_f1", mode="max", dirpath=ckpt_dir,
            filename="best-f1-{epoch:03d}-{val_macro_f1:.4f}"
        )
        stopper = EarlyStopping(
            monitor="val_macro_f1", 
            patience=train_hp['early_stop_patience'], 
            mode="max"
        )

        trainer = pl.Trainer(
            # accelerator="gpu", devices=2, strategy="ddp_spawn",
            max_epochs=train_hp['epochs'],
            logger=logger,
            log_every_n_steps=train_hp['log_every_n_steps'],
            callbacks=[ckpt_top1, ckpt_f1, stopper],
            deterministic=True,
        )

        trainer.fit(lit, train_loader, val_loader)
        trainer.test(ckpt_path=ckpt_f1.best_model_path, dataloaders=test_loader)

        # ---------- gather metrics ----------
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        summaries.append({"seed": seed, **metrics})

    # ---------- write per-seed + summary CSV ----------
    df_path = save_path / f"{model_name}.csv"
    pd.DataFrame(summaries).to_csv(df_path, index=False)
    summarise_csv(df_path, df_path.with_name(df_path.stem + "_summary.csv"))
    print(f"✅  Finished {model_name}; results → {df_path}")



# # ---------------------------------------------------------------------------
# # Main entry point
# # ---------------------------------------------------------------------------
# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--root", type=Path, default=Path("/mnt/ssd/nhsg12m"))
#     parser.add_argument("--save_dir", type=Path, default=Path("/mnt/ssd/baseline_result"))
#     parser.add_argument("--subset", default="one-band")
#     parser.add_argument("--models", nargs="*", default=["gcn", "sage", "gat", "gatv2", "gin", "gine"],
#                         help="Architectures to run; omit for all.")
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--batch_size", type=int, default=256)
#     parser.add_argument("--dim_gnn", type=int, default=128)
#     parser.add_argument("--dim_mlp", type=int, default=1024)
#     parser.add_argument("--layers_gnn", type=int, default=4)
#     parser.add_argument("--layers_mlp", type=int, default=2)
#     parser.add_argument("--heads", type=int, default=1)
#     parser.add_argument("--dropout", type=float, default=0.1)
#     parser.add_argument("--lr_init", type=float, default=1e-3, help="Initial learning rate")
#     parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
#     parser.add_argument("--t0", type=int, default=10, help="T_0 for CosineAnnealingWarmRestarts")
#     parser.add_argument("--t_mult", type=int, default=4, help="T_mult for CosineAnnealingWarmRestarts")
#     parser.add_argument("--seeds", nargs="*", type=int, default=[42, 624, 706])
#     parser.add_argument("--log_every_n_steps", type=int, default=1)
#     parser.add_argument("--early_stop_patience", type=int, default=10)
#     args = parser.parse_args()

#     # sanity: check save_dir exists
#     save_path = os.path.join(args.save_dir, args.subset)
#     os.makedirs(save_path, exist_ok=True)

#     # sanity: if user passes "all", restore full list
#     if args.models == ["all"]:
#         args.models = ["gcn", "sage", "gat", "gatv2", "gin", "gine"]

#     print(f"Loading PolyGraph dataset from {args.root}, subset {args.subset}, "
#             f"batch size {args.batch_size}, models {args.models}, seeds {args.seeds}")

#     dm = HSGLitDataModule(
#         root=args.root,
#         subset=args.subset,
#         batch_size=args.batch_size,
#         seeds=args.seeds
#     )
#     dm.prepare_data(); dm.setup()

#     # loop over architectures
#     for model_name in args.models:
#         args.model = model_name  # inject into hparams for LightningModule
#         summaries: List[Dict[str, float]] = []
#         for seed_idx, seed in enumerate(args.seeds):
#             pl.seed_everything(seed, workers=True)

#             # dataloaders for this seed
#             train_loader = dm.train_dataloader(seed_idx)
#             val_loader = dm.val_dataloader(seed_idx)
#             test_loader = dm.test_dataloader(seed_idx)

#             num_classes = dm.datasets[seed_idx][0].dataset.num_classes
#             in_dim = dm.datasets[seed_idx][0].dataset.num_node_features
#             print(f"Auto detecting {num_classes} classes, {in_dim} input features")

#             lit = LitGNN(args, num_classes, in_dim)

#             logger = TensorBoardLogger(os.path.join(save_path, "tb_logs", model_name), 
#                                        name=f"{model_name}_seed{seed}")
#             model_ckpt_path = os.path.join(save_path, model_name, f"seed_{seed}")
#             ckpt_top1 = ModelCheckpoint(monitor="val_top1", mode="max", dirpath=model_ckpt_path,
#                                         filename="best-top1-{epoch:03d}-{val_top1:.4f}")
#             ckpt_f1 = ModelCheckpoint(monitor="val_macro_f1", mode="max", dirpath=model_ckpt_path,
#                                       filename="best-f1-{epoch:03d}-{val_macro_f1:.4f}")
#             stopper = EarlyStopping(monitor="val_macro_f1", patience=args.early_stop_patience, mode="max")

#             trainer = pl.Trainer(
#                 max_epochs=args.epochs,
#                 logger=logger,
#                 log_every_n_steps=args.log_every_n_steps,
#                 callbacks=[ckpt_top1, ckpt_f1, stopper],
#             )
#             trainer.fit(lit, train_loader, val_loader)
#             trainer.test(ckpt_path=ckpt_f1.best_model_path, dataloaders=test_loader)

#             metrics = trainer.callback_metrics
#             summaries.append({
#                 "seed": seed,
#                 **{k: float(metrics[k]) for k in metrics}
#             })

#         # dump per‑seed + summary CSV
#         out_csv = os.path.join(save_path, f"{model_name}.csv")
#         pd.DataFrame(summaries).to_csv(out_csv, index=False)
#         summarise_csv(out_csv, out_csv.replace(".csv", "_summary.csv"))


# if __name__ == "__main__":
#     main()