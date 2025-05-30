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

# --- Globálne Konštanty pre Augmentácie (Normalizovaný Vstup [-1,1]) ---
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0 

# ------------------------------------------------
# Funkcie na výpočet Štatistík z TRÉNINGOVÉHO setu
# ------------------------------------------------
def get_input_min_max_stats(file_list, data_type_name="Input Data"):
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

def get_target_mean_std_stats(file_list, data_type_name="Target Data"):
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
# Helper Transform Class
# ------------------------------------------------
class AddGaussianNoiseTransform(nn.Module):
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
# Kontrola integrity datasetu
# ------------------------------------------------
def check_dataset_integrity(dataset_path):
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
# Dataset
# ------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, path_to_data, 
                 input_processing_type, 
                 norm_stats_input,      
                 norm_stats_target,     
                 augmentation_strength, 
                 is_train_set=False,
                 target_img_size=(512,512)): 
        
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff"))) # Uprav vzor podľa potreby
        
        self.input_processing_type = input_processing_type
        self.input_min, self.input_max = (None, None) # Inicializácia
        if input_processing_type == 'direct_minmax':
            if norm_stats_input is None:
                raise ValueError("norm_stats_input (min,max) musí byť poskytnutý pre 'direct_minmax'.")
            self.input_min, self.input_max = norm_stats_input

        self.target_mean, self.target_std = norm_stats_target
        
        self.is_train_set = is_train_set
        self.target_img_size = target_img_size
        self.augmentation_strength = augmentation_strength # Uložíme pre referenciu

        self.geometric_transforms = None
        self.pixel_transforms = None

        if self.is_train_set and self.augmentation_strength != 'none':
            self._setup_augmentations(self.augmentation_strength)
        # Ak je augmentation_strength == 'none' alebo is_train_set == False,
        # self.geometric_transforms a self.pixel_transforms zostanú None.

    def _setup_augmentations(self, strength):
        p_affine, deg_affine, trans_affine, scale_affine, shear_affine = 0.0, (-5,5), (0.05,0.05), (0.95,1.05), (-3,3)
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0
        #  _sigma, blur_p = (0.1, 1.0), 0.0 # Parameter pre blur už nie je potrebný

        if strength == 'light':
            p_affine = 0.4 # Tento parameter už nebude mať priamy vplyv na RandomAffine, ale ponechávam pre konzistenciu, ak by ste ho chceli použiť na niečo iné
            noise_std_range, noise_p = (0.02, 0.08), 0.4
            erase_scale, erase_p = (0.01, 0.05), 0.3
            # blur_sigma, blur_p = (0.1, 1.0), 0.3 # Parameter pre blur už nie je potrebný
        elif strength == 'medium':
            p_affine = 0.5; deg_affine,trans_affine,scale_affine,shear_affine = (-10,10),(0.08,0.08),(0.9,1.1),(-5,5) # Parametre pre Affine už nebudú priamo použité
            noise_std_range, noise_p = (0.03, 0.12), 0.5
            erase_scale, erase_p = (0.02, 0.08), 0.4
            # blur_sigma, blur_p = (0.1, 1.8), 0.4 # Parameter pre blur už nie je potrebný
        elif strength == 'strong':
            p_affine = 0.6; deg_affine,trans_affine,scale_affine,shear_affine = (-12,12),(0.1,0.1),(0.85,1.15),(-7,7) # Parametre pre Affine už nebudú priamo použité
            noise_std_range, noise_p = (0.05, 0.15), 0.6
            erase_scale, erase_p = (0.02, 0.10), 0.5
            # blur_sigma, blur_p = (0.1, 2.5), 0.5 # Parameter pre blur už nie je potrebný
        
        geo_transforms_list = [T.RandomHorizontalFlip(p=0.5)]
        # Nasledujúci blok pre RandomAffine je odstránený
        # if p_affine > 0:
        #     fill_value_affine = 0.0 
        #     if self.input_processing_type == 'sincos':
        #         fill_value_affine = [0.0, 0.0] 
        #     geo_transforms_list.append(T.RandomAffine(degrees=deg_affine, translate=trans_affine, 
        #                                              scale=scale_affine, shear=shear_affine, 
        #                                              p=p_affine, fill=fill_value_affine))
        if geo_transforms_list: # Vytvoríme Compose len ak nie je prázdny (RandomHorizontalFlip tam je vždy)
            self.geometric_transforms = T.Compose(geo_transforms_list)
        # else: self.geometric_transforms zostane None (nemalo by nastať, keďže HFlip tam je)
        
        pixel_aug_list = []
        if noise_p > 0:
            pixel_aug_list.append(AddGaussianNoiseTransform(std_dev_range=noise_std_range, p=noise_p,
                                          clamp_min=NORMALIZED_INPUT_CLAMP_MIN,
                                          clamp_max=NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0:
            erase_value = NORMALIZED_INPUT_ERASING_VALUE
            if self.input_processing_type == 'sincos':
                erase_value = [NORMALIZED_INPUT_ERASING_VALUE, NORMALIZED_INPUT_ERASING_VALUE]
            pixel_aug_list.append(T.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3), 
                                value=erase_value, inplace=False))
        # Nasledujúci blok pre GaussianBlur je odstránený
        # if blur_p > 0:
        #     k_size = 3 if isinstance(blur_sigma, float) and blur_sigma <= 1.0 else 5 
        #     pixel_aug_list.append(T.RandomApply([
        #             T.GaussianBlur(kernel_size=k_size, sigma=blur_sigma)
        #         ], p=blur_p))
        
        if pixel_aug_list: # Vytvoríme Compose len ak nie je prázdny
            self.pixel_transforms = T.Compose(pixel_aug_list)
        # else: self.pixel_transforms zostane None (ak nie sú žiadne pixel aug.)

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0
    def _normalize_target_z_score(self, data, mean_val, std_val):
        if std_val < 1e-6: return data - mean_val 
        return (data - mean_val) / std_val
    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        img_numpy = img_numpy.astype(dtype) 
        if img_numpy.shape[-2:] != target_shape: # Kontroluje len H, W dimenzie
            # Jednoduchá logika pre orezanie alebo padding na cieľovú veľkosť
            # Ak potrebuješ robustnejší resize, uprav túto časť
            # Pre teraz, ak rozmery nesedia, vyhodí chybu
            # Tento prístup predpokladá, že tvoje obrázky sú už VÄČŠINOU správnej veľkosti
            # alebo že jednoduchý center crop / reflect pad je akceptovateľný.
            
            # Ak je menší, padneme
            h, w = img_numpy.shape[-2:]
            target_h, target_w = target_shape
            
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                if img_numpy.ndim == 2:
                    img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                elif img_numpy.ndim == 3: # Predpoklad (C, H, W)
                    img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            
            # Ak je väčší, orežeme (center crop)
            h, w = img_numpy.shape[-2:]
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3:
                    img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]

            if img_numpy.shape[-2:] != target_shape: # Finálna kontrola
                 raise ValueError(f"{data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' má tvar {img_numpy.shape} po úprave, očakáva sa H,W ako {target_shape}")
        return img_numpy

    def __len__(self): return len(self.image_list)
    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path, lbl_path = self.image_list[index], self.image_list[index].replace('images','labels').replace('wrappedbg','unwrapped')
        try:
            wrapped_orig_phase, unwrapped_orig = tiff.imread(img_path), tiff.imread(lbl_path)
        except Exception as e: print(f"CHYBA načítania: {img_path} alebo {lbl_path}. Error: {e}"); return None,None
        
        wrapped_orig_phase = self._ensure_shape_and_type(wrapped_orig_phase, self.target_img_size, "Wrapped phase")
        unwrapped_orig = self._ensure_shape_and_type(unwrapped_orig, self.target_img_size, "Unwrapped phase")

        if self.input_processing_type == 'sincos':
            sin_phi = np.sin(wrapped_orig_phase)
            cos_phi = np.cos(wrapped_orig_phase)
            wrapped_input_numpy = np.stack([sin_phi, cos_phi], axis=0)
            wrapped_input_tensor = torch.from_numpy(wrapped_input_numpy.copy())
        elif self.input_processing_type == 'direct_minmax':
            wrapped_tensor_orig = torch.from_numpy(wrapped_orig_phase.copy())
            wrapped_norm_minmax = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor_orig, self.input_min, self.input_max)
            wrapped_input_tensor = wrapped_norm_minmax.unsqueeze(0)
        else: raise ValueError(f"Neznámy input_processing_type: {self.input_processing_type}")
        
        unwrapped_tensor_orig = torch.from_numpy(unwrapped_orig.copy())
        unwrapped_norm_zscore = self._normalize_target_z_score(unwrapped_tensor_orig, self.target_mean, self.target_std)
        unwrapped_target_tensor = unwrapped_norm_zscore.unsqueeze(0) 

        # Augmentácia len pre tréning A AK sú transformácie definované (t.j. augmentation_strength != 'none')
        if self.is_train_set:
            if self.geometric_transforms:
                # Pre tuple vstupov, torchvision.transforms.v2 by to malo zvládnuť
                wrapped_input_tensor, unwrapped_target_tensor = self.geometric_transforms(wrapped_input_tensor, unwrapped_target_tensor)
            if self.pixel_transforms: 
                wrapped_input_tensor = self.pixel_transforms(wrapped_input_tensor)
        
        return wrapped_input_tensor, unwrapped_target_tensor

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch)) # Kontrolujeme len prvý prvok, keďže druhý by mal byť vždy OK ak prvý je
    if not batch: return None, None 
    return torch.utils.data.dataloader.default_collate(batch)

def mae_loss_on_normalized(p, t): return torch.mean(torch.abs(p-t))
def pixel_mse_loss(p, t): return F.mse_loss(p,t)
def sobel_gradient_loss(yt,yp,d):
    if yt.ndim==3: yt=yt.unsqueeze(1)
    if yp.ndim==3: yp=yp.unsqueeze(1)
    sx_w = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=d).view(1,1,3,3)
    sy_w = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=d).view(1,1,3,3)
    gx_t=F.conv2d(yt,sx_w,padding=1); gy_t=F.conv2d(yt,sy_w,padding=1)
    gx_p=F.conv2d(yp,sx_w,padding=1); gy_p=F.conv2d(yp,sy_w,padding=1)
    return torch.mean(torch.abs(gx_t-gx_p)) + torch.mean(torch.abs(gy_t-gy_p))
def denormalize_target_z_score(dn, om, os): return dn*os+om if os>1e-6 else torch.full_like(dn,om)

def run_training_session(
    run_id, device, num_epochs, train_loader, val_loader, test_loader,
    target_original_mean, target_original_std, input_original_min_max,
    encoder_name, encoder_weights, input_processing_type, loss_type, lambda_gdl,
    learning_rate, weight_decay, # Odstránené: scheduler_patience, scheduler_factor
    cosine_T_max, cosine_eta_min, # Pridané pre CosineAnnealingLR
    min_lr, # min_lr sa teraz použije pre cosine_eta_min, ak nie je špecifikované inak
    early_stopping_patience, augmentation_strength ):
    
    config_save_path = f'config_{run_id}.txt'
    config_details = {
        "Run ID": run_id, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Encoder Name": encoder_name, "Encoder Weights": encoder_weights,
        "Input Processing": input_processing_type, "Augmentation Strength": augmentation_strength,
        "Target Norm (Z-score) Mean": f"{target_original_mean:.4f}", 
        "Target Norm (Z-score) Std": f"{target_original_std:.4f}",
        "Loss Type": loss_type, "Lambda GDL": lambda_gdl if 'gdl' in loss_type else "N/A",
        "Initial LR": learning_rate, "Batch Size": train_loader.batch_size, 
        "Num Epochs": num_epochs, "Weight Decay": weight_decay,
        # Odstránené: "Scheduler Patience", "Scheduler Factor"
        "Scheduler Type": "CosineAnnealingLR",
        "CosineAnnealingLR T_max": cosine_T_max,
        "CosineAnnealingLR eta_min": cosine_eta_min,
        "EarlyStopping Patience": early_stopping_patience, "Device": str(device),
    }
    if input_processing_type == 'direct_minmax' and input_original_min_max:
        config_details["Input Norm (MinMax) Min"] = f"{input_original_min_max[0]:.4f}"
        config_details["Input Norm (MinMax) Max"] = f"{input_original_min_max[1]:.4f}"
    
    with open(config_save_path, 'w') as f:
        f.write("Experiment Configuration:\n" + "="*25 + "\n" + 
                "\n".join([f"{k}: {v}" for k,v in config_details.items()]) + "\n")
    print(f"Konfigurácia experimentu uložená do: {config_save_path}")

    in_channels = 1 if input_processing_type == 'direct_minmax' else 2
    net = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                   in_channels=in_channels, classes=1, activation=None).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Nahradenie ReduceLROnPlateau za CosineAnnealingLR
    # Ak cosine_eta_min nie je explicitne nastavené v cfg, použijeme min_lr z pôvodnej konfigurácie
    effective_eta_min = cosine_eta_min if cosine_eta_min is not None else min_lr 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=effective_eta_min)

    train_loss_hist, val_loss_hist, train_mae_denorm_hist, val_mae_denorm_hist = [],[],[],[]
    best_val_mae_denorm, epochs_no_improve = float('inf'), 0
    weights_path = f'best_weights_{run_id}.pth'
    print(f"Starting training for {run_id}...")

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()
        epoch_train_loss_n, epoch_train_mae_d = [], []
        for batch_data in train_loader:
            if batch_data[0] is None: continue
            inputs_n, targets_n = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            preds_n = net(inputs_n)
            main_loss = mae_loss_on_normalized(preds_n, targets_n) if 'mae' in loss_type else pixel_mse_loss(preds_n, targets_n)
            total_loss = main_loss + (lambda_gdl * sobel_gradient_loss(targets_n, preds_n, device) if 'gdl' in loss_type else 0)
            total_loss.backward(); optimizer.step()
            with torch.no_grad():
                preds_d = denormalize_target_z_score(preds_n, target_original_mean, target_original_std)
                targets_d = denormalize_target_z_score(targets_n, target_original_mean, target_original_std)
                epoch_train_mae_d.append(torch.mean(torch.abs(preds_d - targets_d)).item())
            epoch_train_loss_n.append(total_loss.item())
        
        train_loss_hist.append(np.mean(epoch_train_loss_n) if epoch_train_loss_n else float('nan'))
        train_mae_denorm_hist.append(np.mean(epoch_train_mae_d) if epoch_train_mae_d else float('nan'))

        net.eval()
        epoch_val_loss_n, epoch_val_mae_d = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None: continue
                inputs_n, targets_n = batch_data[0].to(device), batch_data[1].to(device)
                preds_n = net(inputs_n)
                main_loss_v = mae_loss_on_normalized(preds_n, targets_n) if 'mae' in loss_type else pixel_mse_loss(preds_n, targets_n)
                total_loss_v = main_loss_v + (lambda_gdl * sobel_gradient_loss(targets_n, preds_n, device) if 'gdl' in loss_type else 0)
                preds_d = denormalize_target_z_score(preds_n, target_original_mean, target_original_std)
                targets_d = denormalize_target_z_score(targets_n, target_original_mean, target_original_std)
                epoch_val_loss_n.append(total_loss_v.item())
                epoch_val_mae_d.append(torch.mean(torch.abs(preds_d - targets_d)).item())

        avg_val_loss_n = np.mean(epoch_val_loss_n) if epoch_val_loss_n else float('nan')
        avg_val_mae_d = np.mean(epoch_val_mae_d) if epoch_val_mae_d else float('nan')
        val_loss_hist.append(avg_val_loss_n)
        val_mae_denorm_hist.append(avg_val_mae_d)

        print(f"Run: {run_id} | Ep {epoch+1}/{num_epochs} | "
              f"Tr L(N): {train_loss_hist[-1]:.4f}, Tr MAE(D): {train_mae_denorm_hist[-1]:.4f} | "
              f"Val L(N): {avg_val_loss_n:.4f}, Val MAE(D): {avg_val_mae_d:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {(time.time()-start_time):.2f}s")

        if not np.isnan(avg_val_mae_d) and avg_val_mae_d < best_val_mae_denorm:
            best_val_mae_denorm = avg_val_mae_d
            torch.save(net.state_dict(), weights_path); print(f"  New best Val MAE(D): {best_val_mae_denorm:.4f}. Saved.")
            epochs_no_improve = 0
        elif not np.isnan(avg_val_mae_d):
            epochs_no_improve += 1; print(f"  Val MAE(D) not improved for {epochs_no_improve} epochs.")
        if epochs_no_improve >= early_stopping_patience: print(f"Early stopping @ epoch {epoch+1}."); break
        
        # scheduler.step() sa volá bez ohľadu na metriku pre CosineAnnealingLR
        scheduler.step()

    print(f"Training of {run_id} done. Best Val MAE(D): {best_val_mae_denorm:.4f} @ {weights_path}")
    
    # Testovanie a vizualizácia (skrátené, ale princíp je rovnaký ako predtým)
    # ... (IMPLEMENTUJ PODOBNE AKO V PREDCHÁDZAJÚCEJ VERZII, ALEBO VYNECHAJ PRE RÝCHLOSŤ)
    if os.path.exists(weights_path):
        print(f"\nTesting with best weights for {run_id}...")
        net.load_state_dict(torch.load(weights_path, weights_only=True))
        net.eval()
        test_mae_d_list = []
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if batch_data[0] is None: continue
                inputs_n, targets_n = batch_data[0].to(device), batch_data[1].to(device)
                preds_n = net(inputs_n)
                preds_d = denormalize_target_z_score(preds_n, target_original_mean, target_original_std)
                targets_d = denormalize_target_z_score(targets_n, target_original_mean, target_original_std)
                test_mae_d_list.extend(torch.mean(torch.abs(preds_d - targets_d), dim=(1,2,3)).cpu().numpy())
                if i == 0 and len(inputs_n)>0: # Vizualizácia prvého obrázku z prvého batchu
                    j=0; pred_img_d=preds_d[j].cpu().numpy().squeeze(); lbl_img_d=targets_d[j].cpu().numpy().squeeze()
                    in_show = inputs_n[j,0,...].cpu().numpy().squeeze() if input_processing_type=='sincos' else inputs_n[j].cpu().numpy().squeeze()
                    vmin_in,vmax_in,title_in = (-1,1,"Input (sin(φ) kanál)") if input_processing_type=='sincos' else (-1,1,"Input (Wrapped, norm. [-1,1])")
                    plt.figure(figsize=(18,6)); plt.suptitle(f"Test Vis - {run_id}",fontsize=16)
                    plt.subplot(1,3,1); plt.imshow(in_show,cmap='gray',vmin=vmin_in,vmax=vmax_in); plt.title(title_in); plt.colorbar()
                    plt.subplot(1,3,2); plt.imshow(lbl_img_d,cmap='gray'); plt.title("GT (denorm.)"); plt.colorbar()
                    plt.subplot(1,3,3); plt.imshow(pred_img_d,cmap='gray'); plt.title("Pred (denorm.)"); plt.colorbar()
                    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(f'vis_{run_id}.png'); plt.close()
                    tiff.imwrite(f'pred_{run_id}.tiff', pred_img_d)

        avg_test_mae_d = np.mean(test_mae_d_list) if test_mae_d_list else float('nan')
        print(f"  Average Test MAE (Denorm): {avg_test_mae_d:.6f}")
        with open(f"metrics_{run_id}.txt", "w") as f:
            f.write(f"Run ID: {run_id}\nBest Val MAE (Denorm): {best_val_mae_denorm:.6f}\nTest MAE (Denorm): {avg_test_mae_d:.6f}\n")
    else: print(f"No weights found for {run_id} to test.")

    plt.figure(figsize=(12,5)); plt.subplot(1,2,1)
    plt.plot(train_loss_hist,label='Train Total Loss(N)'); plt.plot(val_loss_hist,label='Val Total Loss(N)')
    plt.title(f'Total Loss(N) - {run_id}'); plt.xlabel('Epoch'); plt.ylabel('Loss(N)'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(train_mae_denorm_hist,label='Train MAE(D)'); plt.plot(val_mae_denorm_hist,label='Val MAE(D)')
    plt.title(f'MAE(D) - {run_id}'); plt.xlabel('Epoch'); plt.ylabel('MAE(D)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'curves_{run_id}.png'); plt.close()
    
    return {"best_val_mae_denorm": best_val_mae_denorm, "test_mae_denorm": avg_test_mae_d if os.path.exists(weights_path) else float('nan')}


def spusti_experimenty():
    train_base = 'split_dataset_tiff/train_dataset' # UPRAV PODĽA POTREBY
    GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD = get_target_mean_std_stats(
        [p.replace('images','labels').replace('wrappedbg','unwrapped') for p in glob.glob(os.path.join(train_base,'images',"*.tiff")) if os.path.exists(p.replace('images','labels').replace('wrappedbg','unwrapped')) and os.path.exists(p)], "Unwrapped Target")
    GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX = get_input_min_max_stats(
        [p for p in glob.glob(os.path.join(train_base,'images',"*.tiff")) if os.path.exists(p.replace('images','labels').replace('wrappedbg','unwrapped')) and os.path.exists(p)], "Wrapped Input")
    print(f"Globálne štatistiky:\n  Wrapped: Min={GLOBAL_WRAPPED_MIN:.2f}, Max={GLOBAL_WRAPPED_MAX:.2f}\n  Unwrapped: Mean={GLOBAL_UNWRAPPED_MEAN:.2f}, Std={GLOBAL_UNWRAPPED_STD:.2f}")
    if GLOBAL_UNWRAPPED_STD<1e-6: raise ValueError("Std pre unwrapped dáta je nula!")
    if GLOBAL_WRAPPED_MIN==GLOBAL_WRAPPED_MAX: raise ValueError("Min a Max pre wrapped dáta sú rovnaké!")
    
    for ds_name_part in ['train_dataset', 'valid_dataset', 'test_dataset']: # UPRAV CESTY PODĽA POTREBY
        check_dataset_integrity(os.path.join('split_dataset_tiff', ds_name_part))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 0  # Pre jednoduchšie ladenie, potom môžeš zvýšiť na napr. 5

    # --- DEFINÍCIA EXPERIMENTOV ---
    # (Pôvodné run_id_suffix budú prepísané dynamicky generovanými)
    experiments = [
        # --- SÉRIA 1: ResNet34, sin/cos vstup, ladenie GDL a augmentácií ---
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",  
            "input_processing": "direct_minmax", "augmentation_strength": "light",
            "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
            "lr": 1e-3, "bs": 8, "epochs": 50, "es_pat": 50, # sch_pat odstránené
            "cosine_T_max": 50, "cosine_eta_min": 1e-7 # Pridané parametre pre CosineAnnealingLR
        },
        {
            "run_id_suffix": "AUTO_GENERATED", 
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
            "lr": 1e-3, "bs": 8, "epochs": 50, "es_pat": 10,
            "cosine_T_max": 50, "cosine_eta_min": 1e-7
        },
        {
            "run_id_suffix": "AUTO_GENERATED", 
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "medium",
            "loss_type": "mae_gdl", "lambda_gdl": 0.1,
            "lr": 1e-3, "bs": 8, "epochs": 50, "es_pat": 10,
            "cosine_T_max": 25, "cosine_eta_min": 5e-8 # Iné hodnoty pre ukážku
        },
   
        {
            "run_id_suffix": "AUTO_GENERATED", 
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.5,
            "lr": 1e-3, "bs": 8, "epochs": 50, "es_pat": 10,
            "cosine_T_max": 50, "cosine_eta_min": 1e-7
        },

        # --- SÉRIA 2: ResNet34, priamy wrapped vstup, ladenie GDL ---
        # (Porovnanie s sin/cos)
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.3,
            "lr": 1e-3, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 80, "cosine_eta_min": 1e-7
        },
        {
            "run_id_suffix": "AUTO_GENERATED", 
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.3,
            "lr": 1e-4, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 80, "cosine_eta_min": 1e-8 
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "light",
            "loss_type": "mae_gdl", "lambda_gdl": 0.3,
            "lr": 1e-4, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 40, "cosine_eta_min": 1e-7 # T_max môže byť menšie ako epochs
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.5,
            "lr": 1e-4, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 80, "cosine_eta_min": 0 # eta_min môže byť 0
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong",
            "loss_type": "mae_gdl", "lambda_gdl": 0.1,
            "lr": 1e-4, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 80, "cosine_eta_min": 1e-7
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "light",
            "loss_type": "mae_gdl", "lambda_gdl": 0.3,
            "lr": 1e-3, "bs": 8, "epochs": 80, "es_pat": 15,
            "cosine_T_max": 80, "cosine_eta_min": 1e-7
        },
    ]


    all_run_results = []
    for cfg_original in experiments:
        cfg = cfg_original.copy() # Pracujeme s kópiou, aby sme nemenili pôvodný zoznam (ak by to bolo dôležité)

        # --- Dynamické generovanie run_id_suffix ---
        encoder_short_map = {
            "resnet34": "R34", "resnet18": "R18", "resnet50": "R50",
            "mobilenet_v2": "MNv2", "efficientnet-b0": "EffB0",
            # ... doplňte ďalšie podľa potreby ...
        }
        enc_name_part = encoder_short_map.get(cfg["encoder_name"], cfg["encoder_name"])
        
        enc_weights_part = ""
        if cfg["encoder_weights"] is None or (isinstance(cfg["encoder_weights"], str) and cfg["encoder_weights"].lower() == "none"):
            enc_weights_part = "scratch"
        elif isinstance(cfg["encoder_weights"], str) and cfg["encoder_weights"].lower() == "imagenet":
            enc_weights_part = "imgnet"
        
        enc_part = f"{enc_name_part}{enc_weights_part}"


        inp_proc_part = ""
        if cfg["input_processing"] == "sincos":
            inp_proc_part = "sincos"
        elif cfg["input_processing"] == "direct_minmax":
            inp_proc_part = "direct"
        else:
            inp_proc_part = cfg["input_processing"]

        loss_part = ""
        if "mae" in cfg["loss_type"]:
            loss_part = "MAE"
        elif "mse" in cfg["loss_type"]:
            loss_part = "MSE"
        else:
            loss_part = cfg["loss_type"]

        gdl_val = cfg.get("lambda_gdl", 0.0)
        gdl_part = ""
        if gdl_val > 0 and ('gdl' in cfg["loss_type"] or loss_part): # GDL pridáme len ak má zmysel
            gdl_part = f"GDL{str(gdl_val).replace('.', 'p')}"
        
        aug_strength = cfg["augmentation_strength"]
        aug_part = f"Aug{aug_strength.capitalize()}" if aug_strength and aug_strength.lower() != 'none' else "AugNone"
        
        lr_val_str = f"{cfg['lr']:.0e}" 
        lr_part = f"LR{lr_val_str.replace('-', 'm')}"

        # Pridanie ďalších parametrov do suffixu
        epochs_part = f"Ep{cfg['epochs']}"
        es_pat_part = f"ESp{cfg['es_pat']}"
        # sch_pat_part odstránené

        # Nové časti pre CosineAnnealingLR
        cosine_T_max_part = f"Tmax{cfg['cosine_T_max']}"
        # Pre cosine_eta_min, ak je None alebo veľmi malé, môžeme použiť špeciálny formát
        eta_min_val = cfg.get('cosine_eta_min') 
        if eta_min_val is None: # Ak nie je definované, použije sa min_lr z run_training_session
            cosine_eta_min_part = "EtaMinDef" # Default
        elif eta_min_val == 0:
            cosine_eta_min_part = "EtaMin0"
        else:
            cosine_eta_min_part = f"EtaMin{eta_min_val:.0e}".replace('-', 'm')


        # Zostavenie run_id_suffix
        parts = [part for part in [enc_part, inp_proc_part, loss_part, gdl_part, aug_part, lr_part, epochs_part, es_pat_part, cosine_T_max_part, cosine_eta_min_part] if part] 
        cfg['run_id_suffix'] = "_".join(parts)
        # --- Koniec dynamického generovania ---

        print(f"\n\n{'='*25} EXPERIMENT: {cfg['run_id_suffix']} (Pôvodný suffix bol: {cfg_original['run_id_suffix']}) {'='*25}")
        
        train_ds = CustomDataset(f'split_dataset_tiff/train_dataset',
                                 cfg["input_processing"], 
                                 (GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX) if cfg["input_processing"]=='direct_minmax' else None,
                                 (GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD), cfg["augmentation_strength"], True)
        val_ds = CustomDataset(f'split_dataset_tiff/valid_dataset',
                               cfg["input_processing"],
                               (GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX) if cfg["input_processing"]=='direct_minmax' else None,
                               (GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD), 'none', False)
        test_ds = CustomDataset(f'split_dataset_tiff/test_dataset',
                                cfg["input_processing"],
                                (GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX) if cfg["input_processing"]=='direct_minmax' else None,
                                (GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD), 'none', False)

        train_loader = DataLoader(train_ds,batch_size=cfg["bs"],shuffle=True,num_workers=num_workers,collate_fn=collate_fn_skip_none,pin_memory=device.type=='cuda')
        val_loader = DataLoader(val_ds,batch_size=cfg["bs"],shuffle=False,num_workers=num_workers,collate_fn=collate_fn_skip_none,pin_memory=device.type=='cuda')
        test_loader = DataLoader(test_ds,batch_size=cfg["bs"],shuffle=False,num_workers=num_workers,collate_fn=collate_fn_skip_none,pin_memory=device.type=='cuda')
        
        # run_id_final teraz používa dynamicky generovaný cfg['run_id_suffix']
        # a GDL časť je už v ňom, takže ju nemusíme pridávať znova
        run_id_final = f"{cfg['run_id_suffix']}_bs{cfg['bs']}"
        # Pôvodné pridávanie GDL do run_id_final už nie je potrebné, lebo je v suffixe
        # if 'gdl' in cfg['loss_type'] and cfg.get("lambda_gdl",0.0) > 0 : run_id_final+=f"_Lgdl{str(cfg['lambda_gdl']).replace('.','p')}
        
        exp_results = run_training_session(
            run_id=run_id_final, device=device, num_epochs=cfg["epochs"],
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            target_original_mean=GLOBAL_UNWRAPPED_MEAN, target_original_std=GLOBAL_UNWRAPPED_STD,
            input_original_min_max=(GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX) if cfg["input_processing"]=='direct_minmax' else None,
            encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
            input_processing_type=cfg["input_processing"], loss_type=cfg["loss_type"],
            lambda_gdl=cfg.get("lambda_gdl",0.0), learning_rate=cfg["lr"], weight_decay=1e-4, # weight_decay je tu napevno, zvážte pridanie do cfg
            # Odstránené: scheduler_patience=cfg["sch_pat"], scheduler_factor=0.1 , 
            cosine_T_max=cfg["cosine_T_max"], 
            cosine_eta_min=cfg.get("cosine_eta_min"), # .get() pre prípad, že by nebol definovaný
            min_lr=1e-7, # Ponechané pre prípad, že by cosine_eta_min nebol v cfg
            early_stopping_patience=cfg["es_pat"], augmentation_strength=cfg["augmentation_strength"]
        )
        all_run_results.append({"run_id": run_id_final, "config": cfg, "metrics": exp_results})

    print("\n\n" + "="*30 + " SÚHRN VÝSLEDKOV " + "="*30)
    for summary in all_run_results:
        print(f"Run: {summary['run_id']}")
        print(f"  Best Val MAE (D): {summary['metrics'].get('best_val_mae_denorm', 'N/A'):.4f}")
        print(f"  Test MAE (D):     {summary['metrics'].get('test_mae_denorm', 'N/A'):.4f}")
        print("-" * 70)

    print(f"--- VŠETKY EXPERIMENTY DOKONČENÉ ---")

if __name__ == '__main__':
    spusti_experimenty()