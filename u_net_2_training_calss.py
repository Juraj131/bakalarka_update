import os
import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
import tifffile as tiff
import matplotlib.pyplot as plt

from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp

# Wrap count parametre
KMAX = 6
NUM_CLASSES = 2 * KMAX + 1  # = 13 (triedy 0..12)

# ------------------------------------------------
# 1) Kontrola integrity datasetu (images vs labels)
# ------------------------------------------------
def check_dataset_integrity(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels') # V 'labels' sú 'unwrapped' súbory

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    
    # Vytvoríme si zoznam dostupných label súborov pre rýchlejšie overenie
    # Predpokladáme, že label súbory majú názov odvodený od image súborov
    # napr. image: XXX_wrappedbg.tiff -> label: XXX_unwrapped.tiff
    available_label_files = {os.path.basename(f) for f in glob.glob(os.path.join(labels_dir, "*.tiff"))}

    for image_file_path in image_files:
        image_filename = os.path.basename(image_file_path)
        # Odvodíme očakávaný názov label súboru
        expected_label_filename = image_filename.replace('wrappedbg', 'unwrapped')
        
        if expected_label_filename not in available_label_files:
            # Skontrolujeme, či label súbor existuje v plnej ceste, ak by bol problém s basename
            expected_label_path = os.path.join(labels_dir, expected_label_filename)
            if not os.path.exists(expected_label_path):
                 raise FileNotFoundError(f"Label súbor {expected_label_path} pre obrázok {image_file_path} nebol nájdený.")
    print(f"Dataset {dataset_path} je v poriadku.")


# ------------------------------------------------
# 2) Dataset s wrap count + padding (600→608)
# ------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, path_to_data, crop_size=600, augment=False): # augment a crop_size sa tu nepoužívajú
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.k_max = KMAX # Použijeme globálnu KMAX

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        lbl_path = img_path.replace('images', 'labels').replace('wrappedbg', 'unwrapped')

        wrapped = tiff.imread(img_path).astype(np.float32)
        unwrapped = tiff.imread(lbl_path).astype(np.float32)

        wrapped_tensor = torch.tensor(wrapped, dtype=torch.float32).unsqueeze(0)
        unwrapped_tensor = torch.tensor(unwrapped, dtype=torch.float32).unsqueeze(0)

        diff = (unwrapped_tensor - wrapped_tensor) / (2 * np.pi)
        kfloat = torch.round(diff)
        kfloat = torch.clamp(kfloat, -self.k_max, self.k_max)
        klabel = (kfloat + self.k_max).long().squeeze(0) # Shape [H,W]

        wrapped_padded = TF.pad(wrapped_tensor, (4, 4, 4, 4), fill=0)
        unwrapped_padded = TF.pad(unwrapped_tensor, (4, 4, 4, 4), fill=0) # Toto budeme potrebovať pre MSE/MAE
        klabel_padded = TF.pad(klabel, (4, 4, 4, 4), fill=0)

        return wrapped_padded, klabel_padded, unwrapped_padded

# ------------------------------------------------
# 3) Maskované funkcie: CrossEntropy + Accuracy
# ------------------------------------------------
def masked_ce_loss(logits, klabels):
    logits_center = logits[:, :, 4:604, 4:604]
    klabels_center = klabels[:, 4:604, 4:604]
    return F.cross_entropy(logits_center, klabels_center)

def masked_accuracy(logits, klabels):
    logits_center = logits[:, :, 4:604, 4:604]
    klabels_center = klabels[:, 4:604, 4:604]
    pred = torch.argmax(logits_center, dim=1)
    correct = (pred == klabels_center).float().mean()
    return correct

# ------------------------------------------------
# Funkcia pre jeden tréningový beh (KLASIFIKÁCIA podľa u_net_classification_bak.py)
# ------------------------------------------------
def run_training_session(encoder_weights_setting, run_id, device, train_loader, val_loader, test_loader, num_epochs):
    print(f"\n--- Starting CLASSIFICATION Training Session: {run_id} (encoder_weights: {encoder_weights_setting}) ---")

    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights_setting,
        in_channels=1,
        classes=NUM_CLASSES
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    # Scheduler podľa u_net_classification_bak.py
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[85, 115, 140], gamma=0.3)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # Namiesto ukladania best_val_acc, ukladáme váhy na konci tréningu,
    # aby sme replikovali správanie _bak.py, kde sa pre testovanie použijú váhy z poslednej epochy.
    weights_final_epoch_path = f'weights_final_epoch_classification_{run_id}.pth'

    print(f"Starting training for {run_id}...")

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        batch_train_losses = []
        batch_train_accs = []

        for it, (wrapped_batch, klabel_batch, _) in enumerate(train_loader): # unwrapped_batch sa v tréningu nepoužíva
            wrapped_batch = wrapped_batch.to(device)
            klabel_batch = klabel_batch.to(device)

            optimizer.zero_grad()
            logits = net(wrapped_batch)
            loss = masked_ce_loss(logits, klabel_batch)
            loss.backward()
            optimizer.step()
            acc_val = masked_accuracy(logits, klabel_batch)
            batch_train_losses.append(loss.item())
            batch_train_accs.append(acc_val.item())
            # print(f"Epoch {epoch+1}/{num_epochs} | Batch {it+1}/{len(train_loader)}", end='\r') # Odstránené pre čistejší log pri dvoch behoch

        avg_train_loss = np.mean(batch_train_losses)
        avg_train_acc = np.mean(batch_train_accs)
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)

        net.eval()
        batch_val_losses = []
        batch_val_accs = []
        with torch.no_grad():
            for wrapped_batch, klabel_batch, _ in val_loader: # unwrapped_batch sa vo validácii nepoužíva
                wrapped_batch = wrapped_batch.to(device)
                klabel_batch = klabel_batch.to(device)
                logits = net(wrapped_batch)
                val_loss = masked_ce_loss(logits, klabel_batch)
                val_acc = masked_accuracy(logits, klabel_batch)
                batch_val_losses.append(val_loss.item())
                batch_val_accs.append(val_acc.item())

        avg_val_loss = np.mean(batch_val_losses)
        avg_val_acc = np.mean(batch_val_accs)
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(avg_val_acc)
        
        scheduler.step()
        epoch_duration = time.time() - start_time
        print(f"Run: {run_id} | Epoch {epoch+1}/{num_epochs} | "
              f"Train CE: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val CE: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f} | "
              f"Time: {epoch_duration:.2f}s")

    # Uloženie váh z poslednej epochy
    torch.save(net.state_dict(), weights_final_epoch_path)
    print(f"Training completed for {run_id}. Final epoch weights saved to '{weights_final_epoch_path}'.")

    # -----------------------------------------------
    # Testovacia fáza (podľa u_net_classification_bak.py, sekcia 5 a 7)
    # -----------------------------------------------
    print(f"\nLoading final epoch weights for {run_id} from {weights_final_epoch_path} for testing...")
    net.load_state_dict(torch.load(weights_final_epoch_path))
    net.eval()

    test_mse_list = [] # MSE na unwrapped fáze
    test_mae_list = [] # MAE na unwrapped fáze
    # Aj keď _bak.py nepoužíva test_loss/acc pre finálne metriky, môžeme ich sledovať pre konzistenciu
    test_ce_loss_list = [] 
    test_k_acc_list = []


    print(f"Evaluating on the test set for {run_id} (physical unwrapped domain & k-label accuracy)...")
    with torch.no_grad():
        for i, (wrapped_batch, klabel_batch, unwrapped_batch_gt) in enumerate(test_loader):
            wrapped_batch = wrapped_batch.to(device)    # [B,1,608,608]
            klabel_batch = klabel_batch.to(device)      # [B,608,608] - ground truth k-labels
            unwrapped_batch_gt = unwrapped_batch_gt.to(device) # [B,1,608,608] - ground truth unwrapped phase

            logits = net(wrapped_batch) # [B,NUM_CLASSES,608,608]
            
            # Metriky pre k-labels
            ce_loss_val = masked_ce_loss(logits, klabel_batch)
            k_acc_val = masked_accuracy(logits, klabel_batch)
            test_ce_loss_list.append(ce_loss_val.item())
            test_k_acc_list.append(k_acc_val.item())

            # Rekonštrukcia unwrapped fázy z predikovaných k-labels
            pred_class = torch.argmax(logits, dim=1) # [B,608,608] - predikované k-labels (0..12)
            k_pred = pred_class.float() - KMAX       # Transformácia na k-values (-6..6)

            wrapped_squeezed = wrapped_batch.squeeze(1) # [B,608,608]
            unwrapped_pred = wrapped_squeezed + (2 * np.pi) * k_pred # [B,608,608]

            # Výpočet MSE/MAE na unwrapped fáze (stredná časť 600x600)
            gt_center = unwrapped_batch_gt[:, 0, 4:604, 4:604]
            pr_center = unwrapped_pred[:, 4:604, 4:604]

            diff = pr_center - gt_center
            mse_vals = torch.mean(diff**2, dim=(-2, -1)) # Priemer cez H, W pre každý prvok v batchi
            mae_vals = torch.mean(torch.abs(diff), dim=(-2, -1))

            test_mse_list.extend(mse_vals.cpu().numpy())
            test_mae_list.extend(mae_vals.cpu().numpy())

            # Vizualizácia pre prvý batch (ako v u_net_classification_bak.py)
            if i == 0:
                b_idx = 0 # Prvá vzorka v batchi
                
                wrapped_center_viz = wrapped_squeezed[b_idx, 4:604, 4:604].cpu().numpy()
                klabel_center_viz = klabel_batch[b_idx, 4:604, 4:604].cpu().numpy() # GT k-labels (0..12)
                kpred_center_viz = pred_class[b_idx, 4:604, 4:604].cpu().numpy()   # Predikované k-labels (0..12)

                plt.figure(figsize=(12, 4))
                plt.suptitle(f"K-Label Visualization - Run: {run_id}", fontsize=14)
                plt.subplot(1, 3, 1); plt.imshow(wrapped_center_viz, cmap='gray'); plt.title("Wrapped (center)"); plt.colorbar()
                plt.subplot(1, 3, 2); plt.imshow(klabel_center_viz, cmap='gray', vmin=0, vmax=NUM_CLASSES - 1); plt.title("k-label GT (center)"); plt.colorbar()
                plt.subplot(1, 3, 3); plt.imshow(kpred_center_viz, cmap='gray', vmin=0, vmax=NUM_CLASSES - 1); plt.title("k-label Pred (center)"); plt.colorbar()
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(f'example_visualization_wrapcount_{run_id}.png')
                plt.close()

                unwrapped_gt_viz = unwrapped_batch_gt[b_idx, 0, 4:604, 4:604].cpu().numpy()
                unwrapped_pred_viz = unwrapped_pred[b_idx, 4:604, 4:604].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.suptitle(f"Unwrapped Phase Visualization - Run: {run_id}", fontsize=14)
                plt.subplot(1, 3, 1); plt.imshow(wrapped_center_viz, cmap='gray'); plt.title("Wrapped (center)"); plt.colorbar()
                plt.subplot(1, 3, 2); plt.imshow(unwrapped_gt_viz, cmap='gray'); plt.title("Unwrapped GT (center)"); plt.colorbar()
                plt.subplot(1, 3, 3); plt.imshow(unwrapped_pred_viz, cmap='gray'); plt.title("Unwrapped Pred (center)"); plt.colorbar()
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(f'example_visualization_unwrapped_{run_id}.png')
                plt.close()
                
                # Uloženie predikovanej k-mapy a unwrapped mapy ako TIFF
                tiff.imwrite(f'example_test_pred_kmap_{run_id}.tiff', kpred_center_viz.astype(np.uint8))
                tiff.imwrite(f'example_test_pred_unwrapped_{run_id}.tiff', unwrapped_pred_viz.astype(np.float32))


    avg_test_mse = np.mean(test_mse_list)
    avg_test_mae = np.mean(test_mae_list)
    avg_test_ce_loss = np.mean(test_ce_loss_list)
    avg_test_k_acc = np.mean(test_k_acc_list)

    print(f"\nTest Results for {run_id} (Classification Approach):")
    print(f"  Average Test k-label CE Loss: {avg_test_ce_loss:.6f}")
    print(f"  Average Test k-label Accuracy: {avg_test_k_acc:.6f}")
    print(f"  Average Test Unwrapped Phase MSE: {avg_test_mse:.6f}")
    print(f"  Average Test Unwrapped Phase MAE: {avg_test_mae:.6f}")

    with open(f"test_eval_metrics_classification_{run_id}.txt", "w") as f:
        f.write(f"Final Test Evaluation Metrics for Classification Run: {run_id}\n")
        f.write(f"Test k-label CE Loss: {avg_test_ce_loss:.6f}\n")
        f.write(f"Test k-label Accuracy: {avg_test_k_acc:.6f}\n")
        f.write(f"Test Unwrapped Phase MSE (600x600): {avg_test_mse:.6f}\n")
        f.write(f"Test Unwrapped Phase MAE (600x600): {avg_test_mae:.6f}\n")

    # Vykreslenie kriviek tréningového priebehu
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train CE Loss", linewidth=2)
    plt.plot(val_loss_history, label="Val CE Loss", linewidth=2)
    plt.title(f"CE Loss per Epoch - {run_id}")
    plt.xlabel("Epoch"); plt.ylabel("CE Loss"); plt.legend(); plt.grid()
    plt.savefig(f'ce_loss_curve_classification_{run_id}.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label="Train k-label Acc", linewidth=2)
    plt.plot(val_acc_history, label="Val k-label Acc", linewidth=2)
    plt.title(f"k-label Accuracy per Epoch - {run_id}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid()
    plt.savefig(f'accuracy_curve_classification_{run_id}.png')
    plt.close()

    print(f"--- Finished CLASSIFICATION Training Session: {run_id} ---")


# ------------------------------------------------
# 4) Main
# ------------------------------------------------
def spusti_vsetky_klasifikacne_treningy():
    print("--- ZAČÍNAM BLOK KLASIFIKAČNÝCH TRÉNINGOV (z u_net_2_training_calss.py) ---")
    # Kontrola datasetu
    check_dataset_integrity('split_dataset_tiff/train_dataset')
    check_dataset_integrity('split_dataset_tiff/valid_dataset')
    check_dataset_integrity('split_dataset_tiff/test_dataset')

    train_dataset = CustomDataset('split_dataset_tiff/train_dataset', crop_size=608, augment=False)
    val_dataset = CustomDataset('split_dataset_tiff/valid_dataset', crop_size=608, augment=False)
    test_dataset = CustomDataset('split_dataset_tiff/test_dataset', crop_size=608, augment=False) 

    # num_workers podľa _bak.py, môžete upraviť podľa vášho systému
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=5, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    num_epochs_global = 150 # Podľa _bak.py

    # Tréning s predtŕénovaným enkodérom
    run_training_session(
        encoder_weights_setting="imagenet", # Použijeme predtŕénované váhy
        run_id="pretrained_encoder_classification",
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_global
    )

    # Tréning s enkodérom od začiatku (bez predtŕénovaných váh)
    run_training_session(
        encoder_weights_setting=None, # Bez predtŕénovaných váh
        run_id="encoder_from_scratch_classification",
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_global
    )
    print("--- VŠETKY KLASIFIKAČNÉ TRÉNINGY DOKONČENÉ (z u_net_2_training_calss.py) ---")

# ------------------------------------------------
# Pôvodný blok pre priame spustenie skriptu
# ------------------------------------------------
if __name__ == '__main__':
    spusti_vsetky_klasifikacne_treningy() # Teraz voláme novú funkciu

