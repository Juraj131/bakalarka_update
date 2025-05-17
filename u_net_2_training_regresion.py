import os
import glob
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp

# ------------------------------------------------
# 1) Kontrola integrity datasetu (images vs labels)
# ------------------------------------------------
def check_dataset_integrity(dataset_path):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.tiff")))

    for image_file in image_files:
        label_file = image_file.replace('images', 'labels').replace('wrappedbg', 'unwrapped')
        if label_file not in label_files:
            raise FileNotFoundError(f"Label pre obrázok {image_file} nebola nájdená.")
    print(f"Dataset {dataset_path} je v poriadku.")


# ------------------------------------------------
# 2) Dataset - bez normalizácie, s paddingom 600→608
# ------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, path_to_data, crop_size=600, augment=False):
        self.path = path_to_data
        self.crop_size = crop_size
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def random_crop(self, image, label):
        # Vypneme reálne orezávanie (vracia pôvodné rozmery).
        return image, label

    def augment_data(self, image, label):
        # Vypnutá augmentácia (ak by si ju chcel, dopíš sem transformácie).
        return image, label

    def __getitem__(self, index):
        img_path = self.image_list[index]
        lbl_path = img_path.replace('images', 'labels').replace('wrappedbg', 'unwrapped')

        wrapped = tiff.imread(img_path).astype(np.float32)
        unwrapped = tiff.imread(lbl_path).astype(np.float32)

        # Prevedieme na torch.Tensor so shape (1,H,W)
        wrapped_tensor = torch.tensor(wrapped, dtype=torch.float32).unsqueeze(0)
        unwrapped_tensor = torch.tensor(unwrapped, dtype=torch.float32).unsqueeze(0)

        # Bez reálneho orezávania a augmentácie
        wrapped_cropped, unwrapped_cropped = self.random_crop(wrapped_tensor, unwrapped_tensor)
        if self.augment:
            wrapped_cropped, unwrapped_cropped = self.augment_data(wrapped_cropped, unwrapped_cropped)

        # Pôvodne 600×600 → pridáme padding (4 z každej strany) na 608×608
        wrapped_padded   = TF.pad(wrapped_cropped, (4, 4, 4, 4), fill=0)
        unwrapped_padded = TF.pad(unwrapped_cropped, (4, 4, 4, 4), fill=0)

        return wrapped_padded, unwrapped_padded


# ------------------------------------------------
# 3) Definícia maskovaných funkcií straty a metriky
# ------------------------------------------------
def masked_mse_loss(pred, target):
    # pred, target shape: (B,1,608,608)
    # Ignorujeme okraj [0:4], [604:608] → berieme stred [4:604]
    pred_center   = pred[..., 4:604, 4:604]
    target_center = target[..., 4:604, 4:604]
    return F.mse_loss(pred_center, target_center)

def masked_mae_metric(pred, target):
    pred_center   = pred[..., 4:604, 4:604]
    target_center = target[..., 4:604, 4:604]
    return torch.mean(torch.abs(pred_center - target_center))

# ------------------------------------------------
# Funkcia pre jeden tréningový beh
# ------------------------------------------------
def run_training_session(encoder_weights_setting, run_id, device, train_loader, val_loader, test_loader, num_epochs):
    print(f"\n--- Starting Training Session: {run_id} (encoder_weights: {encoder_weights_setting}) ---")

    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights_setting,
        in_channels=1,
        classes=1
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    milestones = [85, 115, 140] # Môžete upraviť, ak je num_epochs iné
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.3)

    train_loss_history = []
    val_loss_history   = []
    train_mae_history  = []
    val_mae_history    = []

    best_val_mae = float('inf')
    weights_save_path = f'best_weights_unet_tiff_{run_id}.pth'
    
    print(f"Starting training for {run_id}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- TRÉNING ---
        net.train()
        epoch_train_loss = []
        epoch_train_mae  = []

        for iter_num, (data_batch, lbl_batch) in enumerate(train_loader):
            # print(f"Epoch {epoch+1}/{num_epochs} | Batch {iter_num+1}/{len(train_loader)}", end='\r') # Zmenené 'iter' na 'iter_num'
            data_batch = data_batch.to(device)
            lbl_batch  = lbl_batch.to(device)

            optimizer.zero_grad()
            output = net(data_batch)
            loss = masked_mse_loss(output, lbl_batch)
            loss.backward()
            optimizer.step()
            mae_val = masked_mae_metric(output, lbl_batch)
            epoch_train_loss.append(loss.item())
            epoch_train_mae.append(mae_val.item())

        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_mae  = np.mean(epoch_train_mae)
        train_loss_history.append(avg_train_loss)
        train_mae_history.append(avg_train_mae)

        # --- VALIDÁCIA ---
        net.eval()
        epoch_val_loss = []
        epoch_val_mae  = []
        with torch.no_grad():
            for data_batch, lbl_batch in val_loader:
                data_batch = data_batch.to(device)
                lbl_batch  = lbl_batch.to(device)
                output = net(data_batch)
                val_loss = masked_mse_loss(output, lbl_batch)
                val_mae  = masked_mae_metric(output, lbl_batch)
                epoch_val_loss.append(val_loss.item())
                epoch_val_mae.append(val_mae.item())

        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_mae  = np.mean(epoch_val_mae)
        val_loss_history.append(avg_val_loss)
        val_mae_history.append(avg_val_mae)

        epoch_duration = time.time() - start_time
        print(f"Run: {run_id} | Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f} | "
              f"Time: {epoch_duration:.2f} s")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(net.state_dict(), weights_save_path)
            #print(f"Run: {run_id} | New best validation MAE: {best_val_mae:.4f}. Weights saved to '{weights_save_path}'.")
        
        scheduler.step()

    print(f"Training completed for {run_id}. Best weights saved to '{weights_save_path}'.")

    # -----------------------------------------------
    # Testovacia fáza
    # -----------------------------------------------
    print(f"\nLoading best weights for {run_id} from {weights_save_path} for testing...")
    net.load_state_dict(torch.load(weights_save_path))
    net.eval()

    test_mse = []
    test_mae = []

    print(f"Evaluating on the test set for {run_id}...")
    with torch.no_grad():
        for i, (data_batch, lbl_batch) in enumerate(test_loader):
            data_batch = data_batch.to(device)
            lbl_batch  = lbl_batch.to(device)
            output = net(data_batch)
            output_center = output[..., 4:604, 4:604]
            lbl_center    = lbl_batch[..., 4:604, 4:604]
            mse_vals = torch.mean((output_center - lbl_center)**2, dim=[1,2,3])
            mae_vals = torch.mean(torch.abs(output_center - lbl_center), dim=[1,2,3])
            test_mse.extend(mse_vals.cpu().numpy())
            test_mae.extend(mae_vals.cpu().numpy())

            if i == 0: # Vizualizácia len pre prvý batch
                j = 0
                pred_img = output_center[j].cpu().numpy().squeeze()
                lbl_img  = lbl_center[j].cpu().numpy().squeeze()
                wrapped_show = data_batch[j].cpu().numpy().squeeze()[4:604, 4:604]
                
                tiff.imwrite(f'example_test_output_{run_id}.tiff', pred_img)
                plt.figure(figsize=(12, 4))
                plt.suptitle(f"Test Visualization - Run: {run_id}", fontsize=16)
                plt.subplot(1, 3, 1)
                plt.imshow(wrapped_show, cmap='gray'); plt.title("Input (center)"); plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(lbl_img, cmap='gray'); plt.title("Ground Truth (center)"); plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(pred_img, cmap='gray'); plt.title("Predicted (center)"); plt.colorbar()
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(f'example_visualization_{run_id}.png')
                plt.close()

    avg_test_mse = np.mean(test_mse)
    avg_test_mae = np.mean(test_mae)
    print(f"\nTest Results for {run_id} (no normalization, ignoring padding):")
    print(f"  Average Test MSE: {avg_test_mse:.6f}")
    print(f"  Average Test MAE: {avg_test_mae:.6f}")

    with open(f"test_eval_metrics_{run_id}.txt", "w") as f:
        f.write(f"Final Test Evaluation Metrics for {run_id}\n")
        f.write(f"Test MSE (600x600): {avg_test_mse:.6f}\n")
        f.write(f"Test MAE (600x600): {avg_test_mae:.6f}\n")

    # -----------------------------------------------
    # Vykreslenie kriviek
    # -----------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train MSE Loss', linestyle='-', linewidth=2)
    plt.plot(val_loss_history,   label='Validation MSE Loss', linestyle='-', linewidth=2)
    plt.title(f'MSE Loss per Epoch - {run_id}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid()
    plt.savefig(f'mse_loss_curve_{run_id}.png')
    # plt.show() # Odkomentujte ak chcete zobraziť interaktívne
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_mae_history, label='Train MAE', linestyle='-', linewidth=2)
    plt.plot(val_mae_history,   label='Validation MAE', linestyle='-', linewidth=2)
    plt.title(f'MAE per Epoch - {run_id}')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid()
    plt.savefig(f'mae_curve_{run_id}.png')
    # plt.show() # Odkomentujte ak chcete zobraziť interaktívne
    plt.close()

    print(f"--- Finished Training Session: {run_id} ---")


# ------------------------------------------------
# 4) Main
# ------------------------------------------------
def spusti_vsetky_regresne_treningy():
    print("--- ZAČÍNAM BLOK REGRESNÝCH TRÉNINGOV (z u_net_2_training_regresion.py) ---")
    # Kontrola datasetu
    check_dataset_integrity('split_dataset_tiff/train_dataset')
    check_dataset_integrity('split_dataset_tiff/valid_dataset')
    check_dataset_integrity('split_dataset_tiff/test_dataset')

    # Vytvorenie datasetov (bez normalizácie)
    # Pôvodné obrázky 600×600 -> s paddingom 608×608
    # crop_size v CustomDataset sa aktuálne nepoužíva na orezanie, ale len ako parameter
    train_dataset = CustomDataset('split_dataset_tiff/train_dataset', crop_size=608, augment=False)
    val_dataset   = CustomDataset('split_dataset_tiff/valid_dataset', crop_size=608, augment=False) # Použite validačný dataset
    test_dataset  = CustomDataset('split_dataset_tiff/test_dataset', crop_size=608, augment=False)  # Použite testovací dataset

    # Odporúčanie: Pre val_dataset a test_dataset zvyčajne používate iné časti datasetu
    # Napr. 'split_dataset_tiff/val_dataset' a 'split_dataset_tiff/test_dataset'
    # Tu pre jednoduchosť používam rovnaký 'train_dataset', čo nie je ideálne pre reálne vyhodnotenie.

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=5, pin_memory=True) # Upravil som num_workers
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=5, pin_memory=True) # Upravil som num_workers
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=5, pin_memory=True) # Upravil som num_workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    num_epochs_global = 150 # Môžete nastaviť globálne alebo pre každé volanie zvlášť

    # Tréning s predtŕénovaným enkodérom
    run_training_session(
        encoder_weights_setting="imagenet",
        run_id="pretrained_encoder_regression",
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_global
    )

    # Tréning s enkodérom od začiatku (bez predtŕénovaných váh)
    run_training_session(
        encoder_weights_setting=None,
        run_id="encoder_from_scratch_regression",
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_global
    )

    print("--- VŠETKY REGRESNÉ TRÉNINGY DOKONČENÉ (z u_net_2_training_regresion.py) ---")


# Tento blok zostáva, ak chcete mať možnosť spustiť skript aj samostatne
if __name__ == '__main__':
    spusti_vsetky_regresne_treningy() # Teraz voláme novú funkciu

# **Kľúčové zmeny:**

# 1.  **Nová funkcia `run_training_session`:**
#     *   Obsahuje takmer celú logiku z pôvodného `if __name__ == '__main__':` bloku, ktorá sa týka jedného tréningového behu.
#     *   Prijíma parametre:
#         *   `encoder_weights_setting`: Hodnota pre `encoder_weights` (buď `"imagenet"` alebo `None`).
#         *   `run_id`: Reťazec na identifikáciu behu (napr. `"pretrained_encoder"`, `"encoder_from_scratch"`), ktorý sa použije v názvoch ukladaných súborov a v logoch.
#         *   `device`, `train_loader`, `val_loader`, `test_loader`, `num_epochs`: Ostatné potrebné objekty a parametre.
#     *   Všetky názvy súborov pre ukladanie váh, grafov a metrík teraz obsahujú `run_id`, aby sa neprepisovali.
#     *   Histórie (`train_loss_history`, atď.) sú lokálne pre túto funkciu, takže sa pre každý beh resetujú.
#     *   Model, optimizer a scheduler sa vytvárajú nanovo na začiatku každého volania funkcie.

# 2.  **Úpravy v `if __name__ == '__main__':`:**
#     *   Príprava datasetov, DataLoaderov a `device` zostáva tu, pretože sú spoločné pre oba behy.
#     *   Funkcia `run_training_session` sa volá dvakrát:
#         *   Prvýkrát s `encoder_weights_setting="imagenet"` a `run_id="pretrained_encoder"`.
#         *   Druhýkrát s `encoder_weights_setting=None` a `run_id="encoder_from_scratch"`.
#     *   Upravil som `num_workers` v `DataLoader`och na nižšie hodnoty, ktoré sú často stabilnejšie, najmä na Windows. Môžete si ich prispôsobiť. Pridal som aj `pin_memory=True` pre potenciálne rýchlejší presun dát na GPU.
#     *   Pridal som poznámku, že pre reálne vyhodnotenie by ste mali mať oddelené datasety pre tréning, validáciu a testovanie.

# 3.  **Ukladanie najlepších váh:** Upravil som logiku ukladania váh tak, aby sa ukladali váhy modelu, ktorý dosiahol najlepšiu (najnižšiu) validačnú MAE počas tréningu daného behu.

# 4.  **Oprava premennej v cykle:** Premenoval som `iter` na `iter_num` v tréningovom cykle, aby nekolidovala s vstavanou funkciou `iter()`.

# 5.  **Nová funkcia `spusti_vsetky_regresne_treningy`:**
#     *   Obsahuje logiku z pôvodného bloku `if __name__ == '__main__':` pre regresné tréningy.
#     *   Volá funkciu `check_dataset_integrity` a vytvára datasety a dataloader pre tréning, validáciu a testovanie.
#     *   Volá `run_training_session` dvakrát: raz s predtrénovaným enkodérom a raz s enkodérom učeným od nuly.
#     *   Tento blok je oddelený od hlavného bloku, takže ak spustíte skript samostatne, môžete mať stále možnosť spustiť len regresné tréningy.

