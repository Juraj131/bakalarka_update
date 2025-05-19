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
import torchvision.transforms.v2 as T 
import segmentation_models_pytorch as smp

# Hodnoty pre augmentácie pre vstup normalizovaný na [-1, 1]
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0 

# ------------------------------------------------
# Funkcie na výpočet Štatistík z TRÉNINGOVÉHO setu
# ------------------------------------------------
def get_input_min_max_stats(file_list, data_type_name="Input Data"): # Pre wrapped vstup
    if not file_list: raise ValueError(f"Prázdny zoznam súborov pre {data_type_name}.")
    min_val_g, max_val_g = np.inf, -np.inf
    print(f"Počítam Min/Max pre {data_type_name} z {len(file_list)} súborov...")
    for i, fp in enumerate(file_list):
        try:
            img = tiff.imread(fp).astype(np.float32)
            min_val_g, max_val_g = min(min_val_g, img.min()), max(max_val_g, img.max())
            if (i+1)%500==0 or (i+1)==len(file_list): print(f"  Spracovaných {i+1}/{len(file_list)}...")
        except Exception as e: print(f"Chyba pri {fp}: {e}")
    if np.isinf(min_val_g) or np.isinf(max_val_g): raise ValueError(f"Nepodarilo sa načítať Min/Max pre {data_type_name}.")
    print(f"Výpočet Min/Max pre {data_type_name} dokončený.")
    return min_val_g, max_val_g

def get_target_mean_std_stats(file_list, data_type_name="Target Data"): # Pre unwrapped cieľ
    # ... (bez zmeny, premenované pre jasnosť) ...
    if not file_list: raise ValueError(f"Prázdny zoznam súborov pre {data_type_name}.")
    all_vals_for_stats = []
    print(f"Počítam Priemer/Std pre {data_type_name} z {len(file_list)} súborov...")
    for i, fp in enumerate(file_list):
        try:
            img = tiff.imread(fp).astype(np.float32)
            all_vals_for_stats.append(img.flatten())
            if (i+1)%500==0 or (i+1)==len(file_list): print(f"  Spracovaných {i+1}/{len(file_list)}...")
        except Exception as e: print(f"Chyba pri {fp}: {e}")
    if not all_vals_for_stats: raise ValueError(f"Nepodarilo sa načítať dáta pre Priemer/Std pre {data_type_name}.")
    cat_vals = np.concatenate(all_vals_for_stats)
    mean_g, std_g = np.mean(cat_vals), np.std(cat_vals)
    print(f"Výpočet Priemer/Std pre {data_type_name} dokončený.")
    return mean_g, std_g


# ------------------------------------------------
# Helper Transform Class (bez zmeny)
# ------------------------------------------------
class AddGaussianNoiseTransform(nn.Module):
    # ... (bez zmeny) ...
    def __init__(self, std_dev_range=(0.03, 0.12), p=0.5, clamp_min=None, clamp_max=None):
        super().__init__()
        self.std_dev_min, self.std_dev_max = std_dev_range
        self.p = p
        self.clamp_min, self.clamp_max = clamp_min, clamp_max

    def forward(self, img_tensor):
        if torch.rand(1).item() < self.p:
            std_dev = torch.empty(1).uniform_(self.std_dev_min, self.std_dev_max).item()
            noise = torch.randn_like(img_tensor) * std_dev
            noisy_img = img_tensor + noise
            if self.clamp_min is not None and self.clamp_max is not None:
                noisy_img = torch.clamp(noisy_img, self.clamp_min, self.clamp_max)
            return noisy_img
        return img_tensor

# ------------------------------------------------
# Kontrola integrity datasetu (bez zmeny)
# ------------------------------------------------
def check_dataset_integrity(dataset_path):
    # ... (bez zmeny) ...
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    if not os.path.exists(labels_dir): raise FileNotFoundError(f"Adresár s labelmi {labels_dir} nebol nájdený.")
    for image_file in image_files:
        label_file_name = os.path.basename(image_file).replace('wrappedbg', 'unwrapped')
        expected_label_path = os.path.join(labels_dir, label_file_name)
        if not os.path.exists(expected_label_path):
            alternative_label_path = image_file.replace('images', 'labels').replace('wrappedbg', 'unwrapped')
            if not os.path.exists(alternative_label_path):
                raise FileNotFoundError(f"Label pre obrázok {image_file} nebola nájdená.")
    print(f"Dataset {dataset_path} je v poriadku.")

# ------------------------------------------------
# Dataset - priamy wrapped vstup (MinMax [-1,1]), cieľ Z-score
# ------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, path_to_data, 
                 input_min_max, # (min,max) pre vstup
                 target_mean_std, # (mean,std) pre cieľ
                 augment=False, is_train_set=False): 
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        
        self.input_min, self.input_max = input_min_max # Pre wrapped vstup
        self.target_mean, self.target_std = target_mean_std # Pre unwrapped cieľ
        
        self.augment = augment
        self.is_train_set = is_train_set
        self.target_img_size = (512, 512)

        self.geometric_transforms = None
        self.pixel_transforms = None

        if self.augment and self.is_train_set:
            self.geometric_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                #T.RandomAffine(degrees=(-10, 10), translate=(0.08, 0.08), scale=(0.9, 1.1), p=0.5)
            ])
            # Augmentácie sa aplikujú na vstup normalizovaný MinMax na [-1,1]
            self.pixel_transforms = T.Compose([
                AddGaussianNoiseTransform(std_dev_range=(0.03, 0.12), p=0.5,
                                          clamp_min=NORMALIZED_INPUT_CLAMP_MIN,
                                          clamp_max=NORMALIZED_INPUT_CLAMP_MAX),
                T.RandomErasing(p=0.4, scale=(0.02, 0.8), ratio=(0.3, 3.3), 
                                value=NORMALIZED_INPUT_ERASING_VALUE, # Pre 1-kanálový vstup
                                inplace=False),
                # T.RandomApply([
                #     T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
                # ], p=0.4) 
            ])

    def _normalize_input_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _normalize_target_z_score(self, data, mean_val, std_val):
        if std_val < 1e-6: 
            print(f"VAROVANIE: Nízka Std ({std_val}) pre normalizáciu cieľa.")
            return data - mean_val 
        return (data - mean_val) / std_val

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        # ... (bez zmeny z predchádzajúcej verzie) ...
        img_numpy = img_numpy.astype(dtype) 
        if img_numpy.shape[-2:] != target_shape:
            raise ValueError(f"{data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' má tvar {img_numpy.shape}, očakáva sa H,W ako {target_shape}")
        return img_numpy

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path = self.image_list[index]
        lbl_path = img_path.replace('images', 'labels').replace('wrappedbg', 'unwrapped')

        try:
            wrapped_orig_phase = tiff.imread(img_path)
            unwrapped_orig = tiff.imread(lbl_path)
        except Exception as e:
            print(f"CHYBA pri načítaní: {img_path} alebo {lbl_path}. Error: {e}")
            return None, None 

        wrapped_orig_phase = self._ensure_shape_and_type(wrapped_orig_phase, self.target_img_size, "Wrapped phase")
        unwrapped_orig = self._ensure_shape_and_type(unwrapped_orig, self.target_img_size, "Unwrapped phase")

        wrapped_tensor_orig = torch.from_numpy(wrapped_orig_phase.copy())
        unwrapped_tensor_orig = torch.from_numpy(unwrapped_orig.copy())

        # Normalizácia VSTUPU na [-1, 1]
        wrapped_input_tensor_norm = self._normalize_input_to_minus_one_one(wrapped_tensor_orig, self.input_min, self.input_max)
        wrapped_input_tensor = wrapped_input_tensor_norm.unsqueeze(0) # (1, H, W)
        
        # Normalizácia CIEĽA na Z-score
        unwrapped_norm_zscore = self._normalize_target_z_score(unwrapped_tensor_orig, self.target_mean, self.target_std)
        unwrapped_target_tensor = unwrapped_norm_zscore.unsqueeze(0) # (1, H, W)

        if self.augment and self.is_train_set:
            if self.geometric_transforms:
                # Pre jednokanálový vstup a cieľ to T.Compose zvládne
                wrapped_input_tensor, unwrapped_target_tensor = self.geometric_transforms(wrapped_input_tensor, unwrapped_target_tensor)
            if self.pixel_transforms: 
                wrapped_input_tensor = self.pixel_transforms(wrapped_input_tensor) # Pixel aug len na vstup
        
        return wrapped_input_tensor, unwrapped_target_tensor

# Koláčová funkcia pre DataLoader (bez zmeny)
def collate_fn_skip_none(batch):
    # ... (bez zmeny) ...
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch:
        return None, None 
    return torch.utils.data.dataloader.default_collate(batch)

# ------------------------------------------------
# Loss Funkcie a Denormalizácia (bez zmeny)
# ------------------------------------------------
# ... (mae_loss_on_normalized, sobel_gradient_loss, denormalize_target_z_score - bez zmeny) ...
def mae_loss_on_normalized(pred_norm, target_norm): 
    return torch.mean(torch.abs(pred_norm - target_norm))

def sobel_gradient_loss(y_true_norm, y_pred_norm, device):
    if y_true_norm.ndim == 3: y_true_norm = y_true_norm.unsqueeze(1)
    if y_pred_norm.ndim == 3: y_pred_norm = y_pred_norm.unsqueeze(1)
    sobel_x_weights = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y_weights = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    g_true_x = F.conv2d(y_true_norm, sobel_x_weights, padding=1)
    g_true_y = F.conv2d(y_true_norm, sobel_y_weights, padding=1)
    g_pred_x = F.conv2d(y_pred_norm, sobel_x_weights, padding=1)
    g_pred_y = F.conv2d(y_pred_norm, sobel_y_weights, padding=1)
    loss = torch.mean(torch.abs(g_true_x - g_pred_x)) + \
           torch.mean(torch.abs(g_true_y - g_pred_y))
    return loss

def denormalize_target_z_score(data_norm, original_mean, original_std):
    if original_std < 1e-6: return torch.full_like(data_norm, original_mean)
    return data_norm * original_std + original_mean

# ------------------------------------------------
# Tréningová Slučka
# ------------------------------------------------
def run_training_session(encoder_name_str, encoder_weights_setting, run_id, device, 
                         train_loader, val_loader, test_loader, num_epochs,
                         target_original_mean, target_original_std, # Pre denorm. cieľa
                         input_original_min, input_original_max,   # Pre denorm. vstupu pri vizualizácii
                         lambda_gdl=0.1):
    print(f"\n--- Starting Training Session (Direct Wrapped Input): {run_id} ---")

    net = smp.Unet(
        encoder_name=encoder_name_str,
        encoder_weights=encoder_weights_setting,
        in_channels=1, # ZMENA: Vstup je teraz 1-kanálový
        classes=1,
        activation=None 
    ).to(device)

    # ... (optimizer, scheduler, histórie, best_val_mae, early_stopping - bez zásadných zmien) ...
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-7)

    train_loss_history_norm = [] 
    val_loss_history_norm = []
    train_mae_history_denorm = [] 
    val_mae_history_denorm   = []

    best_val_mae_denorm = float('inf')
    weights_save_path = f'best_weights_{run_id}.pth'
    
    early_stopping_patience = 20 
    epochs_no_improve = 0
    
    print(f"Starting training for {run_id}...")

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        epoch_train_total_loss_norm_list = []
        epoch_train_mae_denorm_list  = []
        for iter_num, batch_data in enumerate(train_loader):
            if batch_data[0] is None: 
                print(f"Preskakujem chybný batch v tréningu, iterácia {iter_num}")
                continue
            # data_batch_norm je teraz priamy wrapped normalizovaný vstup
            data_batch_input_norm, lbl_batch_norm = batch_data 
            data_batch_input_norm = data_batch_input_norm.to(device)
            lbl_batch_norm  = lbl_batch_norm.to(device)
            optimizer.zero_grad()
            output_norm = net(data_batch_input_norm) # Sieť spracuje 1-kanálový vstup
            
            mae_loss_norm = mae_loss_on_normalized(output_norm, lbl_batch_norm)
            gdl_loss_norm_val = sobel_gradient_loss(lbl_batch_norm, output_norm, device=device)
            total_loss_norm = mae_loss_norm + lambda_gdl * gdl_loss_norm_val
            total_loss_norm.backward()
            optimizer.step()
            
            with torch.no_grad():
                output_denorm = denormalize_target_z_score(output_norm, target_original_mean, target_original_std)
                lbl_denorm    = denormalize_target_z_score(lbl_batch_norm, target_original_mean, target_original_std)
                mae_denorm_val = torch.mean(torch.abs(output_denorm - lbl_denorm))
                
            epoch_train_total_loss_norm_list.append(total_loss_norm.item())
            epoch_train_mae_denorm_list.append(mae_denorm_val.item())

        avg_train_total_loss_norm = np.mean(epoch_train_total_loss_norm_list) if epoch_train_total_loss_norm_list else float('nan')
        avg_train_mae_denorm  = np.mean(epoch_train_mae_denorm_list) if epoch_train_mae_denorm_list else float('nan')
        train_loss_history_norm.append(avg_train_total_loss_norm)
        train_mae_history_denorm.append(avg_train_mae_denorm)

        # --- VALIDÁCIA --- (podobné úpravy ako v tréningu)
        net.eval()
        epoch_val_total_loss_norm_list = []
        epoch_val_mae_denorm_list  = []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None:
                    print(f"Preskakujem chybný batch vo validácii")
                    continue
                data_batch_input_norm, lbl_batch_norm = batch_data
                data_batch_input_norm = data_batch_input_norm.to(device)
                lbl_batch_norm  = lbl_batch_norm.to(device)
                output_norm = net(data_batch_input_norm)
                
                mae_loss_norm_val = mae_loss_on_normalized(output_norm, lbl_batch_norm)
                gdl_loss_norm_val = sobel_gradient_loss(lbl_batch_norm, output_norm, device=device)
                val_total_loss_norm_val = mae_loss_norm_val + lambda_gdl * gdl_loss_norm_val

                output_denorm = denormalize_target_z_score(output_norm, target_original_mean, target_original_std)
                lbl_denorm    = denormalize_target_z_score(lbl_batch_norm, target_original_mean, target_original_std)
                val_mae_denorm_val  = torch.mean(torch.abs(output_denorm - lbl_denorm))
                
                epoch_val_total_loss_norm_list.append(val_total_loss_norm_val.item())
                epoch_val_mae_denorm_list.append(val_mae_denorm_val.item())
 
        avg_val_total_loss_norm = np.mean(epoch_val_total_loss_norm_list) if epoch_val_total_loss_norm_list else float('nan')
        avg_val_mae_denorm  = np.mean(epoch_val_mae_denorm_list) if epoch_val_mae_denorm_list else float('nan')
        val_loss_history_norm.append(avg_val_total_loss_norm)
        val_mae_history_denorm.append(avg_val_mae_denorm)

        epoch_duration = time.time() - start_time
        print(f"Run: {run_id} | Epoch {epoch+1}/{num_epochs} | "
              f"Train TotalLoss(N): {avg_train_total_loss_norm:.4f}, MAE(D): {avg_train_mae_denorm:.4f} | "
              f"Val TotalLoss(N): {avg_val_total_loss_norm:.4f}, MAE(D): {avg_val_mae_denorm:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {epoch_duration:.2f} s")

        current_val_metric_to_monitor = avg_val_mae_denorm
        # ... (zvyšok early stopping a scheduler logiky bez zmeny) ...
        if not np.isnan(current_val_metric_to_monitor) and current_val_metric_to_monitor < best_val_mae_denorm:
            best_val_mae_denorm = current_val_metric_to_monitor
            torch.save(net.state_dict(), weights_save_path)
            print(f"Run: {run_id} | New best validation MAE (Denorm): {best_val_mae_denorm:.4f}. Weights saved.")
            epochs_no_improve = 0
        elif not np.isnan(current_val_metric_to_monitor):
            epochs_no_improve += 1
            print(f"Run: {run_id} | Val MAE (Denorm) did not improve for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stopping_patience: 
            print(f"Early stopping triggered after {epoch+1} epochs for run {run_id}.")
            break
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not np.isnan(current_val_metric_to_monitor):
                 scheduler.step(current_val_metric_to_monitor)
        else:
            scheduler.step()
    
    # ... (zvyšok funkcie: dokončenie tréningu, testovacia fáza, ploty) ...
    # V testovacej fáze a vizualizácii tiež použi data_batch_input_norm
    # a prípadne denormalizuj vstup pre vizualizáciu, ak chceš vidieť pôvodné hodnoty wrapped fázy.
    print(f"Training completed for {run_id}. Best Val MAE (Denorm): {best_val_mae_denorm:.4f}. Weights saved to '{weights_save_path}'.")

    if not os.path.exists(weights_save_path):
        print(f"VAROVANIE: Súbor s váhami {weights_save_path} neexistuje! Testovanie bude preskočené.")
    else:
        print(f"\nLoading best weights for {run_id} from {weights_save_path} for testing...")
        net.load_state_dict(torch.load(weights_save_path, weights_only=True))
        net.eval()
        test_mae_denorm_list = []
        print(f"Evaluating on the test set for {run_id}...")
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if batch_data[0] is None:
                    print(f"Preskakujem chybný batch v teste, iterácia {i}")
                    continue
                data_batch_input_norm, lbl_batch_norm = batch_data # Vstup je teraz priamy wrapped normalizovaný
                data_batch_input_norm = data_batch_input_norm.to(device)
                lbl_batch_norm  = lbl_batch_norm.to(device)
                output_norm = net(data_batch_input_norm)
                
                output_denorm = denormalize_target_z_score(output_norm, target_original_mean, target_original_std)
                lbl_denorm    = denormalize_target_z_score(lbl_batch_norm, target_original_mean, target_original_std)
                mae_per_sample = torch.mean(torch.abs(output_denorm - lbl_denorm), dim=(1,2,3))
                test_mae_denorm_list.extend(mae_per_sample.cpu().numpy())

                if i == 0 and len(data_batch_input_norm) > 0:
                    j = 0
                    pred_img_denorm = output_denorm[j].cpu().numpy().squeeze()
                    lbl_img_denorm  = lbl_denorm[j].cpu().numpy().squeeze()
                    # Vstupný obrázok je normalizovaný na [-1,1]
                    wrapped_show_norm = data_batch_input_norm[j].cpu().numpy().squeeze() 
                    
                    tiff.imwrite(f'example_test_output_denorm_direct_{run_id}.tiff', pred_img_denorm)
                    plt.figure(figsize=(18, 6))
                    plt.suptitle(f"Test Visualization (Direct Wrapped Input) - Run: {run_id}", fontsize=16)
                    plt.subplot(1, 3, 1)
                    im1 = plt.imshow(wrapped_show_norm, cmap='gray', vmin=-1, vmax=1);
                    plt.title("Input (Wrapped, norm. to [-1,1])") 
                    plt.colorbar(im1)
                    plt.subplot(1, 3, 2)
                    plt.imshow(lbl_img_denorm, cmap='gray'); plt.title("GT (denorm.)") 
                    plt.colorbar()
                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_img_denorm, cmap='gray'); plt.title("Predicted (denorm.)") 
                    plt.colorbar()
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig(f'example_visualization_denorm_direct_{run_id}.png')
                    plt.close()

        avg_test_mae_denorm = np.mean(test_mae_denorm_list) if test_mae_denorm_list else float('nan')
        print(f"\nTest Results for {run_id} (Denormalized MAE):")
        print(f"  Average Test MAE (Denorm): {avg_test_mae_denorm:.6f}")
        with open(f"test_eval_metrics_denorm_direct_{run_id}.txt", "w") as f:
            f.write(f"Final Test Evaluation Metrics for {run_id} (Direct Wrapped Input)\n")
            f.write(f"Test MAE (Denorm): {avg_test_mae_denorm:.6f}\n")

    # Grafy (bez zásadných zmien, len názvy súborov)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history_norm, label='Train Total Loss (Normalized)', linestyle='-', linewidth=2)
    plt.plot(val_loss_history_norm,   label='Validation Total Loss (Normalized)', linestyle='-', linewidth=2)
    plt.title(f'Total Combined Loss (Normalized) per Epoch - {run_id}')
    plt.xlabel('Epoch'); plt.ylabel('Loss (Normalized)'); plt.legend(); plt.grid(True)
    plt.savefig(f'loss_norm_direct_curve_{run_id}.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_mae_history_denorm, label='Train MAE (Denormalized)', linestyle='-', linewidth=2)
    plt.plot(val_mae_history_denorm,   label='Validation MAE (Denormalized)', linestyle='-', linewidth=2)
    plt.title(f'MAE (Denormalized) per Epoch - {run_id}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (Denormalized)'); plt.legend(); plt.grid(True)
    plt.savefig(f'mae_denorm_direct_curve_{run_id}.png')
    plt.close()

    print(f"--- Finished Training Session: {run_id} ---")

# ------------------------------------------------
# Main
# ------------------------------------------------
def spusti_vsetky_regresne_treningy():
    RUN_ID_PREFIX = "direct_wrapped_gdl512_MAE_Zscore_strongAug" 
    ENCODER_NAME = "mobilenet_v2" # Skúsime opäť resnet34 s týmto prístupom

    print(f"--- ZAČÍNAM BLOK REGRESNÝCH TRÉNINGOV ({RUN_ID_PREFIX}) ---")
    train_base_path = 'split_dataset_tiff/train_dataset'
    train_images_path = os.path.join(train_base_path, 'images') 
    train_labels_path = os.path.join(train_base_path, 'labels') 

    train_wrapped_files_for_list = sorted(glob.glob(os.path.join(train_images_path, "*.tiff")))
    if not train_wrapped_files_for_list:
        raise SystemExit(f"V tréningovom adresári {train_images_path} neboli nájdené žiadne wrapped obrázky.")
    train_unwrapped_files = [p.replace('images', 'labels').replace('wrappedbg', 'unwrapped') for p in train_wrapped_files_for_list]
    
    valid_train_unwrapped_files = []
    valid_train_wrapped_files = [] # Potrebujeme aj pre wrapped min/max
    for i, uf_path in enumerate(train_unwrapped_files):
        wf_path = train_wrapped_files_for_list[i]
        if os.path.exists(uf_path) and os.path.exists(wf_path): 
            valid_train_unwrapped_files.append(uf_path)
            valid_train_wrapped_files.append(wf_path)
        else:
            print(f"VAROVANIE: Chýba súbor pre pár: {wf_path} alebo {uf_path}. Tento pár bude preskočený.")
    
    if not valid_train_unwrapped_files or not valid_train_wrapped_files:
        raise SystemExit("Žiadne validné páry súborov nájdené pre výpočet štatistík. Končím.")

    # Štatistiky pre VSTUP (MinMax)
    GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX = get_input_min_max_stats(valid_train_wrapped_files, "Wrapped Input")
    # Štatistiky pre CIEĽ (Z-score)
    GLOBAL_UNWRAPPED_MEAN, GLOBAL_UNWRAPPED_STD = get_target_mean_std_stats(valid_train_unwrapped_files, "Unwrapped Target")

    print(f"Globálne štatistiky pre normalizáciu (z tréningového setu):")
    print(f"  Wrapped Input (MinMax to [-1,1]): Min={GLOBAL_WRAPPED_MIN:.4f}, Max={GLOBAL_WRAPPED_MAX:.4f}")
    print(f"  Unwrapped Target (Z-score): Mean={GLOBAL_UNWRAPPED_MEAN:.4f}, Std={GLOBAL_UNWRAPPED_STD:.4f}")

    if GLOBAL_WRAPPED_MIN == GLOBAL_WRAPPED_MAX: raise ValueError("Min a Max pre wrapped dáta sú rovnaké!")
    if GLOBAL_UNWRAPPED_STD < 1e-6: raise ValueError("Std pre unwrapped dáta je príliš blízko nuly!")

    for ds_name in ['train_dataset', 'valid_dataset', 'test_dataset']:
        check_dataset_integrity(os.path.join('split_dataset_tiff', ds_name))

    norm_input_stats = (GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX)
    norm_target_stats = (GLOBAL_UNWRAPPED_MEAN, GLOBAL_UNWRAPPED_STD)

    train_dataset = CustomDataset(os.path.join('split_dataset_tiff', 'train_dataset'),
                                  input_min_max=norm_input_stats, 
                                  target_mean_std=norm_target_stats,
                                  augment=True, is_train_set=True)
    val_dataset   = CustomDataset(os.path.join('split_dataset_tiff', 'valid_dataset'),
                                  input_min_max=norm_input_stats,
                                  target_mean_std=norm_target_stats,
                                  augment=False, is_train_set=False)
    test_dataset  = CustomDataset(os.path.join('split_dataset_tiff', 'test_dataset'),
                                  input_min_max=norm_input_stats,
                                  target_mean_std=norm_target_stats,
                                  augment=False, is_train_set=False)

    num_workers_setting = 5 
    bs = 8 
    print(f"Používam Batch Size: {bs}")

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,  num_workers=num_workers_setting, 
                              pin_memory=torch.cuda.is_available() and num_workers_setting > 0, collate_fn=collate_fn_skip_none)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=num_workers_setting, 
                              pin_memory=torch.cuda.is_available() and num_workers_setting > 0, collate_fn=collate_fn_skip_none)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False, num_workers=num_workers_setting, 
                              pin_memory=torch.cuda.is_available() and num_workers_setting > 0, collate_fn=collate_fn_skip_none)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda': print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    num_epochs_global = 50
    lambda_gdl_setting = 0.1 
    current_run_id = f"{RUN_ID_PREFIX}_{ENCODER_NAME}_lr1e-4_bs{bs}_lambdagdl{str(lambda_gdl_setting).replace('.', 'p')}"

    run_training_session(
        encoder_name_str=ENCODER_NAME,
        encoder_weights_setting='imagenet',
        run_id=current_run_id,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_global,
        target_original_mean=GLOBAL_UNWRAPPED_MEAN,
        target_original_std=GLOBAL_UNWRAPPED_STD,
        input_original_min=GLOBAL_WRAPPED_MIN, # Pre denormalizáciu vstupu pri vizualizácii
        input_original_max=GLOBAL_WRAPPED_MAX,
        lambda_gdl=lambda_gdl_setting
    )
    print(f"--- VŠETKY REGRESNÉ TRÉNINGY ({current_run_id}) DOKONČENÉ ---")

if __name__ == '__main__':
    spusti_vsetky_regresne_treningy()