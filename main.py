import os
import json
import warnings
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from spdnet.optimizer import StiefelMetaOptimizer
from datasets import Dataset_PPMI
from model import MambaSPD_Attn_then_SSM

warnings.filterwarnings("ignore")


def main():

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_path = '/home/ms/cyw/data/taowu'

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch_size = 4
    num_workers = 0
    num_classes = 2
    k_folds = 10
    learning_rate = 4e-3
    weight_decay = 3e-4
    seed = 1114
    down_dims = [96, 64, 32, 16]
    window_sizes=[16, 32, 64, 96]
    d_state=32
    bottleneck=16*16,

    set_seed(seed)


    all_dataset = Dataset_PPMI(data_path)
    file_paths = all_dataset.data_path

    n_samples = len(all_dataset)
    print(f"dataset len: {n_samples}")

    sample_X, _ = all_dataset[0]
    dim = sample_X.shape[-1]
    model1 = MambaSPD_Attn_then_SSM(
            n=dim,
            window_sizes=window_sizes,
            d_state=d_state,
            bottleneck=32*32,
            num_classes=num_classes,
            down_dims = down_dims,
            metric='logeuclid', stable_ssm=True, use_rbn=True, attn_mode='geo'
        )

    total_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    def train(model, train_loader, optimizer, epoch, criterion):
        model.train()
        total_loss = 0.0
        preds, gts = [], []

        bar = tqdm(train_loader, total=len(train_loader), desc=f"Train Epoch {epoch+1}/{epochs}")
        for Y_t, label in bar:

            Y_t = Y_t.squeeze().to(device).float()
            labels = label.squeeze().to(device)
            if Y_t.dim() == 2:
                labels = labels.unsqueeze(0)

            optimizer.zero_grad()

            outputs= model(Y_t)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            preds.append(pred_labels)
            gts.append(labels.cpu().numpy())
            bar.set_postfix(loss=loss.item())

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(gts, preds)
        avg_pre = precision_score(gts, preds, average='weighted')
        avg_rec = recall_score(gts, preds, average='weighted')
        avg_f1 = f1_score(gts, preds, average='weighted')

        return avg_loss, avg_acc, avg_pre, avg_rec, avg_f1

    @torch.no_grad()
    def test(model, test_loader, epoch, criterion):
        model.eval()
        total_loss = 0.0
        preds, gts, alphas = [], [], []
        count = 0


        bar = tqdm(test_loader, total=len(test_loader), desc=f"Test Epoch {epoch+1}/{epochs}")

        for Y_t, label in bar:
            Y_t = Y_t.squeeze().to(device).float()
            labels = label.squeeze().to(device)
            if Y_t.dim() == 2:
                labels = labels.unsqueeze(0)

            count = count+1

            outputs = model(Y_t)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            preds.append(pred_labels)
            gts.append(labels.cpu().numpy())


            bar.set_postfix(loss=total_loss/count)

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)


        avg_loss = total_loss / len(test_loader)
        avg_acc = accuracy_score(gts, preds)
        avg_pre = precision_score(gts, preds, average='weighted')
        avg_rec = recall_score(gts, preds, average='weighted')
        avg_f1 = f1_score(gts, preds, average='weighted')

        return avg_loss, avg_acc, avg_pre, avg_rec, avg_f1



    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    avg_accs, avg_pres, avg_f1s, avg_recs, avg_alaph, avg_train, avg_test = [], [], [], [], [], [], []
    avg_gpu_mem = []
    model_save_dir = "best_models/taowu"
    os.makedirs(model_save_dir, exist_ok=True)

    result_log = os.path.join(model_save_dir, "taowu.txt")

    with open(result_log, "w") as f:
        f.write(f"Model Init Params: down_dims={down_dims}, window_sizes={window_sizes}, "
                f"d_state={d_state}, bottleneck={bottleneck}, num_classes={num_classes}, "
                f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, wd={weight_decay}\n"
                f"Total number of parameters: {total_params}")
        f.write(f"Dataset: {all_dataset.__class__.__name__}\n")
        f.write(f"K-Folds: {k_folds}, Seed: {seed}, date: {datetime.now()}\n\n")


    for fold, (train_idx, val_idx) in enumerate(kfold.split(file_paths)):

        criterion = nn.CrossEntropyLoss()

        print(f"===== Fold {fold+1}/{k_folds} =====")
        model_path = os.path.join(model_save_dir, f"mambaspd_fold_{fold+1}.pth")

        train_files = [file_paths[i] for i in train_idx]
        val_files = [file_paths[i] for i in val_idx]

        with open(os.path.join(model_save_dir, f"split_fold_{fold+1}.json"), "w") as f:
            json.dump({"train_files": train_files, "val_files": val_files}, f, indent=2)

        g = torch.Generator()
        g.manual_seed(seed)

        train_dataset = Dataset_PPMI(root_dir=data_path, spatial=False, graph=False, mgnn=False, file_paths=train_files)
        val_dataset = Dataset_PPMI(root_dir=data_path, spatial=False, graph=False, mgnn=False, file_paths=val_files)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
            generator=g
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
            generator=g
        )

        sample_X, _ = all_dataset[0]
        dim = sample_X.shape[-1]

        model = MambaSPD_Attn_then_SSM(
            n=dim,
            window_sizes=window_sizes,
            d_state=d_state,
            bottleneck=32*32,
            num_classes=num_classes,
            down_dims = down_dims,
            metric='logeuclid', stable_ssm=True, use_rbn=True, attn_mode='geo'
        )

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = StiefelMetaOptimizer(optimizer)

        best_acc, best_model_state = 0.0, None
        best_pre, best_f1, best_rec = 0.0, 0.0, 0.0

        for epoch in range(epochs):

            train_loss, train_acc, _, _, _ = train(model, train_loader, optimizer, epoch, criterion)
            test_loss, test_acc, test_pre, test_rec, test_f1 = test(model, val_loader, epoch, criterion)

            if test_acc > best_acc:
                best_acc = test_acc
                best_pre = test_pre
                best_f1 = test_f1
                best_rec = test_rec

                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'best_pre': best_pre,
                    'best_f1': best_f1,
                    'best_rec': best_rec,

                }
                print(f"acc: {best_acc}, pre: {best_pre}, f1: {best_f1}, rec: {best_rec}")


        torch.save(best_model_state, model_path)
        avg_accs.append(best_acc)
        avg_pres.append(best_pre)
        avg_f1s.append(best_f1)
        avg_recs.append(best_rec)

        log_line = (f"[Fold {fold+1}] "
                    f"Acc={best_acc:.4f}, Pre={best_pre:.4f}, "
                    f"F1={best_f1:.4f}, Rec={best_rec:.4f}")

        print(log_line)
        with open(result_log, "a") as f:
            f.write(log_line + "\n")

        import gc
        del train_loader, val_loader, train_dataset, val_dataset, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    final_line = (f"Cross-validation "
                  f"Acc={np.mean(avg_accs):.4f}±{np.std(avg_accs):.4f}, "
                  f"Pre={np.mean(avg_pres):.4f}±{np.std(avg_pres):.4f}, "
                  f"F1={np.mean(avg_f1s):.4f}±{np.std(avg_f1s):.4f}, "
                  f"Rec={np.mean(avg_recs):.4f}±{np.std(avg_recs):.4f}")
    print(final_line)
    with open(result_log, "a") as f:
        f.write(final_line + "\n")

if __name__ == "__main__":
    main()
