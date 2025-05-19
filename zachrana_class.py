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

# --- Globálne Konštanty ---
KMAX = 6 # Maximálny počet |2pi| skokov na klasifikáciu
NUM_CLASSES = 2 * KMAX + 1  # Počet tried pre k-labely (napr. 13 pre KMAX=6)

# Hodnoty pre augmentácie pre vstup normalizovaný na [-1, 1]
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0 

# --- Štatistické Funkcie ---
def get_input_min_max_stats(file_list, data_type_name="Input Data"):
    # ... (bez zmeny z predchádzajúceho multifunkčného kódu) ...
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

# --- Augmentačná Trieda ---
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

# --- Kontrola Integrity ---
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

# --- Dataset pre Klasifikáciu Wrap Count ---
class WrapCountDataset(Dataset):
    def __init__(self, path_to_data, 
                 input_min_max, # (min,max) pre normalizáciu wrapped vstupu
                 k_max_val=KMAX,
                 augment_strength='none', # 'none', 'light', 'medium', 'strong'
                 is_train_set=False,
                 target_img_size=(512,512)):
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        self.input_min, self.input_max = input_min_max
        self.k_max = k_max_val
        self.is_train_set = is_train_set
        self.target_img_size = target_img_size
        self.augment_strength = augment_strength

        self.geometric_transforms = None
        self.pixel_transforms_for_input = None # Len pre wrapped vstup

        if self.is_train_set and self.augment_strength != 'none':
            self._setup_augmentations(self.augment_strength)

    def _setup_augmentations(self, strength):
        # ... (definície p_affine, deg_affine, atď. zostávajú rovnaké) ...
        p_affine, deg_affine, trans_affine, scale_affine, shear_affine = 0.0, (-5,5), (0.05,0.05), (0.95,1.05), (-3,3)
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0
        blur_sigma, blur_p = (0.1, 1.0), 0.0

        if strength == 'light':
            p_affine = 0.3
            noise_std_range, noise_p = (0.02, 0.08), 0.3
            erase_scale, erase_p = (0.01, 0.05), 0.2
            blur_sigma, blur_p = (0.1, 1.0), 0.2
        elif strength == 'medium':
            p_affine = 0.5; deg_affine,trans_affine,scale_affine,shear_affine = (-10,10),(0.08,0.08),(0.9,1.1),(-5,5)
            noise_std_range, noise_p = (0.03, 0.12), 0.5
            erase_scale, erase_p = (0.02, 0.08), 0.4
            blur_sigma, blur_p = (0.1, 1.8), 0.4
        elif strength == 'strong':
            p_affine = 0.6; deg_affine,trans_affine,scale_affine,shear_affine = (-12,12),(0.1,0.1),(0.85,1.15),(-7,7)
            noise_std_range, noise_p = (0.05, 0.15), 0.6
            erase_scale, erase_p = (0.02, 0.10), 0.5
            blur_sigma, blur_p = (0.1, 2.5), 0.5
        
        geo_transforms_list = [T.RandomHorizontalFlip(p=0.5)] # RandomHorizontalFlip má parameter 'p'
        if p_affine > 0:
            # Pre wrapped vstup (1 kanál) normalizovaný na [-1,1], fill=0.0 je stred
            # Ak by si používal sin/cos (2 kanály), fill by mal byť [0.0, 0.0]
            # V tomto klasifikačnom kóde je vstup 1-kanálový (normalizovaný wrapped)
        #     affine_transform = T.RandomAffine(degrees=deg_affine, translate=trans_affine, 
        #                                       scale=scale_affine, shear=shear_affine, 
        #                                       fill=0.0) # fill pre jednokanálový vstup
        #     # Obalíme RandomAffine do RandomApply
        #     geo_transforms_list.append(T.RandomApply([affine_transform], p=p_affine))
            
        # self.geometric_transforms = T.Compose(geo_transforms_list)
            pass
        
        pixel_aug_list = []
        if noise_p > 0:
            pixel_aug_list.append(AddGaussianNoiseTransform(std_dev_range=noise_std_range, p=noise_p,
                                          clamp_min=NORMALIZED_INPUT_CLAMP_MIN,
                                          clamp_max=NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0:
            pixel_aug_list.append(T.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3), 
                                value=NORMALIZED_INPUT_ERASING_VALUE, inplace=False))
        if blur_p > 0:
            # k_size = 3 if isinstance(blur_sigma, float) and blur_sigma <= 1.0 else 5 
            # pixel_aug_list.append(T.RandomApply([
            #         T.GaussianBlur(kernel_size=k_size, sigma=blur_sigma)
            #     ], p=blur_p))
            pass
        
        if pixel_aug_list: self.pixel_transforms_for_input = T.Compose(pixel_aug_list)
        else: self.pixel_transforms_for_input = T.Compose([])

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        # ... (rovnaká robustná implementácia ako predtým) ...
        img_numpy = img_numpy.astype(dtype) 
        if img_numpy.shape[-2:] != target_shape:
            raise ValueError(f"{data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' má tvar {img_numpy.shape}, očakáva sa H,W ako {target_shape}")
        return img_numpy

    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path, lbl_path = self.image_list[index], self.image_list[index].replace('images','labels').replace('wrappedbg','unwrapped')
        try:
            wrapped_orig = tiff.imread(img_path) 
            unwrapped_orig = tiff.imread(lbl_path)
        except Exception as e: print(f"CHYBA načítania: {img_path} alebo {lbl_path}. Error: {e}"); return None,None,None

        wrapped_orig = self._ensure_shape_and_type(wrapped_orig, self.target_img_size, "Wrapped phase")
        unwrapped_orig = self._ensure_shape_and_type(unwrapped_orig, self.target_img_size, "Unwrapped phase")

        wrapped_tensor = torch.from_numpy(wrapped_orig.copy()).unsqueeze(0) # (1, H, W)
        unwrapped_tensor = torch.from_numpy(unwrapped_orig.copy()).unsqueeze(0) # (1, H, W)

        # Aplikácia geometrických augmentácií na oba tenzory PRED výpočtom k-labelov
        if self.is_train_set and self.augment_strength != 'none' and self.geometric_transforms:
            # Pre konzistentnú transformáciu ich stackneme (ak by transformácie nezvládali tuple priamo)
            # Alebo ak sú transformácie v T.Compose([....]) z v2, mali by zvládnuť tuple
            wrapped_tensor, unwrapped_tensor = self.geometric_transforms(wrapped_tensor, unwrapped_tensor)

        # Výpočet k-labelov z (potenciálne augmentovaných) wrapped a unwrapped
        diff = (unwrapped_tensor - wrapped_tensor) / (2 * np.pi)
        k_float = torch.round(diff)
        k_float = torch.clamp(k_float, -self.k_max, self.k_max)
        # Výsledný k_label bude (H,W) a typu long pre CrossEntropy
        k_label = (k_float + self.k_max).long().squeeze(0) 

        # Normalizácia wrapped vstupu (teraz už môže byť augmentovaný geometricky)
        # .squeeze(0) a .unsqueeze(0) aby sme normalizovali 2D dáta a vrátili 3D
        wrapped_input_norm = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor.squeeze(0), self.input_min, self.input_max).unsqueeze(0)

        # Aplikácia pixel-wise augmentácií len na normalizovaný wrapped vstup
        if self.is_train_set and self.augment_strength != 'none' and self.pixel_transforms_for_input:
            wrapped_input_norm = self.pixel_transforms_for_input(wrapped_input_norm)
        
        # unwrapped_tensor sa vracia pre výpočet finálnych MAE/MSE na denormalizovaných dátach
        return wrapped_input_norm, k_label, unwrapped_tensor.squeeze(0) # Vstup (1,H,W), k_label (H,W), unwrapped (H,W)

# Koláčová funkcia - upravená pre 3 výstupy z datasetu
def collate_fn_skip_none_classification(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, batch))
    if not batch: return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Loss Funkcie a Metriky pre Klasifikáciu ---
def cross_entropy_loss_full(logits, klabels): # Bez maskovania, ak už nemáme padding
    return F.cross_entropy(logits, klabels) # logits (B,C,H,W), klabels (B,H,W)

def k_label_accuracy_full(logits, klabels): # Bez maskovania
    pred_classes = torch.argmax(logits, dim=1) # (B,H,W)
    correct = (pred_classes == klabels).float().sum() # Počet správnych pixelov
    total = klabels.numel() # Celkový počet pixelov
    return correct / total

# --- Tréningová Slučka pre Klasifikáciu ---
def run_classification_training_session(
    run_id, device, num_epochs, train_loader, val_loader, test_loader,
    input_min_max, # Pre info do configu, nepoužíva sa na denormalizáciu (len unwrapped)
    # Model
    encoder_name, encoder_weights, k_max_val,
    # Optimizer a Scheduler
    learning_rate, weight_decay, scheduler_patience, scheduler_factor, min_lr,
    # Early Stopping
    early_stopping_patience,
    # Augmentácie
    augmentation_strength
    ):

    config_save_path = f'config_clf_{run_id}.txt'
    # ... (uloženie configu podobne ako predtým, pridaj k_max_val) ...
    config_details = {
        "Run ID": run_id, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Task Type": "Classification (Wrap Count)",
        "Encoder Name": encoder_name, "Encoder Weights": encoder_weights,
        "K_MAX": k_max_val, "NUM_CLASSES": 2 * k_max_val + 1,
        "Input Normalization": f"MinMax to [-1,1] (Min: {input_min_max[0]:.2f}, Max: {input_min_max[1]:.2f})",
        "Augmentation Strength": augmentation_strength,
        "Initial LR": learning_rate, "Batch Size": train_loader.batch_size, 
        "Num Epochs": num_epochs, "Weight Decay": weight_decay,
        "Scheduler Patience": scheduler_patience, "Scheduler Factor": scheduler_factor, "Min LR": min_lr,
        "EarlyStopping Patience": early_stopping_patience, "Device": str(device),
    }
    with open(config_save_path, 'w') as f:
        f.write("Experiment Configuration:\n" + "="*25 + "\n" + 
                "\n".join([f"{k}: {v}" for k,v in config_details.items()]) + "\n")
    print(f"Konfigurácia klasifikačného experimentu uložená do: {config_save_path}")


    num_classes_effective = 2 * k_max_val + 1
    net = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                   in_channels=1, classes=num_classes_effective, activation=None).to(device) # Výstup sú logity

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', # Chceme maximalizovať Val k-label Accuracy
                                                     factor=scheduler_factor, patience=scheduler_patience, 
                                                     verbose=True, min_lr=min_lr)
    
    train_ce_loss_hist, val_ce_loss_hist = [], []
    train_k_acc_hist, val_k_acc_hist = [], []
    # Pre sledovanie finálnej MAE na unwrapped fáze
    train_final_mae_hist, val_final_mae_hist = [], []


    best_val_k_accuracy = 0.0 # Chceme maximalizovať
    weights_path = f'best_weights_clf_{run_id}.pth'
    epochs_no_improve = 0
    print(f"Starting CLASSIFICATION training for {run_id}...")

    for epoch in range(num_epochs):
        start_time = time.time() # Začiatok merania času epochy
        net.train()
        epoch_train_ce, epoch_train_k_acc, epoch_train_final_mae = [], [], []
        for batch_data in train_loader:
            if batch_data[0] is None: continue
            wrapped_norm, k_labels, unwrapped_gt_orig = batch_data # unwrapped_gt_orig je nenormalizovaný
            wrapped_norm, k_labels = wrapped_norm.to(device), k_labels.to(device)
            unwrapped_gt_orig = unwrapped_gt_orig.to(device) # Pre výpočet finálnej MAE

            optimizer.zero_grad()
            logits = net(wrapped_norm) # (B, NUM_CLASSES, H, W)
            
            loss = cross_entropy_loss_full(logits, k_labels)
            loss.backward(); optimizer.step()
            
            with torch.no_grad():
                acc = k_label_accuracy_full(logits, k_labels)
                # Rekonštrukcia pre finálnu MAE
                pred_classes = torch.argmax(logits, dim=1) # (B,H,W)
                k_pred_values = pred_classes.float() - k_max_val # (-k_max .. +k_max)
                # wrapped_norm je (B,1,H,W), unwrapped_gt_orig je (B,H,W)
                # Potrebujeme denormalizovať wrapped_norm, aby sme ho mohli použiť na rekonštrukciu
                # Ak je wrapped_norm už [-1,1] z [-pi,pi], tak wrapped_orig = (wrapped_norm+1)/2 * 2pi - pi
                # Alebo jednoduchšie, použiť wrapped_orig z datasetu, ak by sme ho vracali
                # Pre jednoduchosť teraz predpokladáme, že wrapped_norm je to, čo by sme použili
                # s tým, že k_pred_values sú už v správnych jednotkách (násobky 2pi)
                # Ak wrapped_norm je [-1,1] z pôvodného wrapped_phase, tak wrapped_phase = (wrapped_norm + 1)/2 * (max_in-min_in) + min_in
                # Toto je komplikovanejšie, jednoduchšie je vrátiť wrapped_orig z datasetu pre MAE.
                # Nateraz, pre ilustráciu, použijeme wrapped_norm, ale je to NESPRÁVNE pre presnú MAE.
                # Potrebovali by sme pôvodný wrapped_orig (nenormalizovaný) z datasetu.
                # Pre finálnu MAE by mal CustomDataset vracať aj pôvodný wrapped_orig.
                # Nateraz to preskočíme a zameriame sa na k-label accuracy.
                # Ak by sme chceli presnú MAE, museli by sme upraviť CustomDataset.
                # Alternatíva: vypočítať MAE na k-labeloch * 2pi (čo nie je presne to isté)
                # mae_on_k_times_2pi = torch.mean(torch.abs(k_pred_values - (k_labels.float() - k_max_val))) * (2 * np.pi)
                # epoch_train_final_mae.append(mae_on_k_times_2pi.item())
                # Tu ponecháme len k_acc a CE loss pre tréning
                
            epoch_train_ce.append(loss.item())
            epoch_train_k_acc.append(acc.item())
        
        # ... (podobná logika pre validáciu - výpočet CE, k_acc a prípadne rekonštrukcia pre MAE)
        # ... (printy, scheduler, early stopping - riadené podľa val_k_acc_hist[-1])

        # Nasledujúci kód je veľmi zjednodušený pre validáciu, treba ho doplniť
        avg_train_ce = np.mean(epoch_train_ce) if epoch_train_ce else float('nan')
        avg_train_k_acc = np.mean(epoch_train_k_acc) if epoch_train_k_acc else float('nan')
        train_ce_loss_hist.append(avg_train_ce)
        train_k_acc_hist.append(avg_train_k_acc)

        # Validácia - zjednodušená
        net.eval()
        epoch_val_ce, epoch_val_k_acc = [],[]
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None: continue
                wrapped_norm, k_labels, _ = batch_data
                wrapped_norm, k_labels = wrapped_norm.to(device), k_labels.to(device)
                logits = net(wrapped_norm)
                epoch_val_ce.append(cross_entropy_loss_full(logits, k_labels).item())
                epoch_val_k_acc.append(k_label_accuracy_full(logits, k_labels).item())
        
        avg_val_ce = np.mean(epoch_val_ce) if epoch_val_ce else float('nan')
        avg_val_k_acc = np.mean(epoch_val_k_acc) if epoch_val_k_acc else float('nan')
        val_ce_loss_hist.append(avg_val_ce)
        val_k_acc_hist.append(avg_val_k_acc)

        # Výpočet času trvania epochy
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Run: {run_id} | Ep {epoch+1}/{num_epochs} | Tr CE: {avg_train_ce:.4f}, Tr kAcc: {avg_train_k_acc:.4f} | Val CE: {avg_val_ce:.4f}, Val kAcc: {avg_val_k_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} | Time: {epoch_duration:.2f}s")

        current_val_metric = avg_val_k_acc # Monitorujeme k-label accuracy
        if not np.isnan(current_val_metric) and current_val_metric > best_val_k_accuracy:
            best_val_k_accuracy = current_val_metric
            torch.save(net.state_dict(), weights_path); print(f"  New best Val k-Acc: {best_val_k_accuracy:.4f}. Saved.")
            epochs_no_improve = 0
        elif not np.isnan(current_val_metric):
            epochs_no_improve += 1; print(f"  Val k-Acc not improved for {epochs_no_improve} epochs.")
        if epochs_no_improve >= early_stopping_patience: print(f"Early stopping @ epoch {epoch+1}."); break
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau) and not np.isnan(current_val_metric): scheduler.step(current_val_metric)
        elif not isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step()
    
    print(f"Training of {run_id} done. Best Val k-Acc: {best_val_k_accuracy:.4f} @ {weights_path}")

    # --- Testovacia Fáza pre Klasifikáciu ---
    # (Implementuj podobne ako v tvojom pôvodnom `u_net_classification_bak.py`
    #  s načítaním `best_weights_clf_{run_id}.pth`, rekonštrukciou unwrapped fáze
    #  a výpočtom finálnej MAE/MSE na unwrapped dátach.
    #  Nezabudni na vizualizácie k-labelov a rekonštruovanej fázy.)
    # TOTO JE LEN PLACEHOLDER PRE TESTOVACIU FÁZU
    avg_test_mae_denorm = float('nan')
    if os.path.exists(weights_path):
        print(f"\nTesting with best weights for {run_id}...")
        net.load_state_dict(torch.load(weights_path, weights_only=True))
        net.eval()
        test_final_mae_list = []
        test_k_acc_list_final = []
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if batch_data[0] is None: continue
                wrapped_norm, k_labels_gt, unwrapped_gt_orig = batch_data
                wrapped_norm, k_labels_gt, unwrapped_gt_orig = wrapped_norm.to(device), k_labels_gt.to(device), unwrapped_gt_orig.to(device)
                
                logits = net(wrapped_norm)
                pred_classes = torch.argmax(logits, dim=1) # (B,H,W)
                k_pred_values = pred_classes.float() - k_max_val

                # Pre presnú rekonštrukciu by sme potrebovali pôvodný wrapped (nenormalizovaný)
                # Ak ho nemáme, musíme denormalizovať wrapped_norm:
                # wrapped_orig_for_reconstruction = (wrapped_norm * (input_min_max[1] - input_min_max[0]) + input_min_max[0] + input_min_max[1]) / 2.0 # Ak bol MinMax na [-1,1]
                # Pre jednoduchosť teraz použijeme wrapped_norm, ale výsledná MAE nebude presne vo fyz. jednotkách bez denormalizácie wrapped
                # Ak CustomDataset vracia wrapped_orig (nenormalizovaný), použi ten.
                
                # Aby sme to urobili správne, CustomDataset musí vracať aj pôvodný wrapped_orig
                # Pre teraz, na ukážku, budeme MAE počítať len na k-hodnotách * 2pi, čo je aproximácia
                # k_gt_values = k_labels_gt.float() - k_max_val
                # mae_approx = torch.mean(torch.abs(k_pred_values - k_gt_values)) * (2 * np.pi)
                # test_final_mae_list.append(mae_approx.item()) # Toto je len aproximácia!

                # Správna rekonštrukcia (vyžaduje pôvodný wrapped, alebo denormalizovaný wrapped_norm)
                # Pre ilustráciu, ak by sme mali pôvodný wrapped (nenormalizovaný)
                # unwrapped_pred_reconstructed = wrapped_orig_denormalized_or_direct + (2 * np.pi) * k_pred_values
                # final_mae_on_reconstructed = torch.mean(torch.abs(unwrapped_pred_reconstructed - unwrapped_gt_orig))
                # test_final_mae_list.append(final_mae_on_reconstructed.item())
                # Pre teraz, keďže CustomDataset nevracia wrapped_orig, test MAE bude len placeholder
                
                test_k_acc_list_final.append(k_label_accuracy_full(logits, k_labels_gt).item())

                if i==0 and len(wrapped_norm)>0: # Vizualizácia
                    j=0
                    # ... (kód pre vizualizáciu k-labelov a rekonštruovanej fázy)
                    # ... (podobne ako si mal v u_net_classification_bak.py)
                    # ... (pre rekonštrukciu budeš potrebovať pôvodný wrapped alebo ho denormalizovať)
                    pass # Placeholder pre vizualizáciu

        avg_test_mae_denorm = np.mean(test_final_mae_list) if test_final_mae_list else float('nan') # Toto bude NaN, ak neimplementuješ rekonštrukciu
        avg_test_k_acc = np.mean(test_k_acc_list_final) if test_k_acc_list_final else float('nan')
        print(f"  Test k-label Accuracy: {avg_test_k_acc:.4f}")
        # print(f"  Approximated Test Unwrapped MAE (Denorm): {avg_test_mae_denorm:.4f}") # Ak by si rátal tú aproximáciu
        with open(f"metrics_clf_{run_id}.txt", "w") as f:
            f.write(f"Run ID: {run_id}\nBest Val k-Acc: {best_val_k_accuracy:.4f}\nTest k-Acc: {avg_test_k_acc:.4f}\n")
            # f.write(f"Approximated Test Unwrapped MAE: {avg_test_mae_denorm:.4f}\n")
    else:
        print(f"No weights found for {run_id} to test.")
        avg_test_mae_denorm = float('nan') # Aby sme mali čo vrátiť

    # Grafy pre klasifikáciu
    plt.figure(figsize=(12,5)); plt.subplot(1,2,1)
    plt.plot(train_ce_loss_hist,label='Train CE Loss'); plt.plot(val_ce_loss_hist,label='Val CE Loss')
    plt.title(f'CE Loss - {run_id}'); plt.xlabel('Epoch'); plt.ylabel('CE Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(train_k_acc_hist,label='Train k-Acc'); plt.plot(val_k_acc_hist,label='Val k-Acc')
    plt.title(f'k-Label Accuracy - {run_id}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'curves_clf_{run_id}.png'); plt.close()

    return {"best_val_k_accuracy": best_val_k_accuracy, "test_final_mae_denorm": avg_test_mae_denorm, "test_k_accuracy": avg_test_k_acc if os.path.exists(weights_path) else float('nan')}


# --- Hlavná Funkcia pre Spúšťanie Experimentov ---
def spusti_experimenty_klasifikacia():
    RUN_ID_PREFIX = "clf_wrapcount_v1" 
    # Predpokladáme, že GLOBAL_WRAPPED_MIN/MAX sú blízko -pi/pi pre MinMax normalizáciu vstupu
    # Ak nie, treba ich vypočítať z tréningových wrapped dát.
    # Pre jednoduchosť teraz použijeme priamo -np.pi, np.pi
    GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX = -np.pi, np.pi 
    norm_input_minmax_stats = (GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX)

    # Cieľové dáta (unwrapped) sa nepoužívajú na normalizáciu siete, len na výpočet k-labelov
    # a na finálnu evaluáciu MAE/MSE na rekonštruovanej fáze.
    # Štatistiky pre unwrapped nepotrebujeme pre `target_mean_std` v CustomDataset.

    for ds_name_part in ['train_dataset', 'valid_dataset', 'test_dataset']:
        check_dataset_integrity(os.path.join('split_dataset_tiff', ds_name_part))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 5
    
    experiments_clf = [
        {
            "run_id_suffix": "ResNet18_K6_AugMedium",
            "encoder_name": "resnet18", "encoder_weights": "imagenet",
            "k_max_val": KMAX, # Použijeme globálnu KMAX
            "augmentation_strength": "strong", # 'none', 'light', 'medium', 'strong'
            "lr": 1e-4, "bs": 8, "epochs": 50, 
            "es_pat": 15, "sch_pat": 7, "sch_factor": 0.1, "min_lr": 1e-7
        },
        # {
        #     "run_id_suffix": "ResNet34_K6_AugStrong",
        #     "encoder_name": "resnet34", "encoder_weights": "imagenet",
        #     "k_max_val": KMAX,
        #     "augmentation_strength": "strong",
        #     "lr": 1e-4, "bs": 8, "epochs": 100, 
        #     "es_pat": 20, "sch_pat": 10, "sch_factor": 0.1, "min_lr": 1e-7
        # },
    ]

    all_clf_results = []
    for cfg in experiments_clf:
        print(f"\n\n{'='*20} KLASIFIKAČNÝ EXPERIMENT: {cfg['run_id_suffix']} {'='*20}")
        
        # Pre klasifikáciu CustomDataset nepotrebuje target_mean_std
        train_ds = WrapCountDataset(f'split_dataset_tiff/train_dataset',
                                 input_min_max=norm_input_minmax_stats,
                                 k_max_val=cfg["k_max_val"],
                                 augment_strength=cfg["augmentation_strength"], 
                                 is_train_set=True)
        val_ds = WrapCountDataset(f'split_dataset_tiff/valid_dataset',
                               input_min_max=norm_input_minmax_stats,
                               k_max_val=cfg["k_max_val"],
                               augment_strength='none', 
                               is_train_set=False)
        test_ds = WrapCountDataset(f'split_dataset_tiff/test_dataset',
                                input_min_max=norm_input_minmax_stats,
                                k_max_val=cfg["k_max_val"],
                                augment_strength='none',
                                is_train_set=False)

        train_loader = DataLoader(train_ds,batch_size=cfg["bs"],shuffle=True,num_workers=num_workers,collate_fn=collate_fn_skip_none_classification,pin_memory=device.type=='cuda')
        val_loader = DataLoader(val_ds,batch_size=cfg["bs"],shuffle=False,num_workers=num_workers,collate_fn=collate_fn_skip_none_classification,pin_memory=device.type=='cuda')
        test_loader = DataLoader(test_ds,batch_size=cfg["bs"],shuffle=False,num_workers=num_workers,collate_fn=collate_fn_skip_none_classification,pin_memory=device.type=='cuda')
        
        run_id_final = f"{cfg['run_id_suffix']}_bs{cfg['bs']}"
        
        exp_results = run_classification_training_session(
            run_id=run_id_final, device=device, num_epochs=cfg["epochs"],
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            input_min_max=norm_input_minmax_stats, # Pre info do configu
            encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
            k_max_val=cfg["k_max_val"],
            learning_rate=cfg["lr"], weight_decay=1e-4,
            scheduler_patience=cfg["sch_pat"], scheduler_factor=cfg.get("sch_factor", 0.2), 
            min_lr=cfg.get("min_lr", 1e-7),
            early_stopping_patience=cfg["es_pat"], 
            augmentation_strength=cfg["augmentation_strength"]
        )
        all_clf_results.append({"run_id": run_id_final, "config": cfg, "metrics": exp_results})

    print("\n\n" + "="*30 + " SÚHRN KLASIFIKAČNÝCH VÝSLEDKOV " + "="*30)
    for summary in all_clf_results:
        print(f"Run: {summary['run_id']}")
        print(f"  Best Val k-Acc: {summary['metrics'].get('best_val_k_accuracy', 'N/A'):.4f}")
        # print(f"  Test Unwrapped MAE (Denorm): {summary['metrics'].get('test_final_mae_denorm', 'N/A'):.4f}")
        print(f"  Test k-Acc: {summary['metrics'].get('test_k_accuracy', 'N/A'):.4f}")
        print("-" * 70)
    print(f"--- VŠETKY KLASIFIKAČNÉ EXPERIMENTY DOKONČENÉ ---")


if __name__ == '__main__':
    # Najprv definuj všetky potrebné triedy a funkcie (AddGaussianNoiseTransform, atď.)
    # ktoré sú hore, alebo ich importuj, ak sú v inom súbore.
    spusti_experimenty_klasifikacia()