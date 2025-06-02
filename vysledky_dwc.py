import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# import torch.optim as optim # Pre testovanie nie je potrebný optimizer
import torch.nn.functional as F
import tifffile as tiff
import matplotlib.pyplot as plt
# import torchvision.transforms.v2 as T # Pre testovanie sa augmentácie zvyčajne nepoužívajú
import segmentation_models_pytorch as smp

# --- Globálne Konštanty (ak sú potrebné) ---
# NORMALIZED_INPUT_CLAMP_MIN = -1.0 # Z dwc.py, ak by bolo potrebné pre nejakú vizualizáciu normalizovaného vstupu
# NORMALIZED_INPUT_CLAMP_MAX = 1.0
KMAX_DEFAULT_FALLBACK = 6 # Fallback, ak K_MAX nie je v configu

# ----------------------------------------------------------------------------------
# KOPÍROVANÉ TRIEDY A FUNKCIE Z dwc.py (alebo ich ekvivalenty)
# ----------------------------------------------------------------------------------

class WrapCountDataset(Dataset): # Prevzaté a upravené z dwc.py pre testovanie
    def __init__(self, path_to_data,
                 input_min_max_global,
                 k_max_val, # K_MAX sa načíta z configu
                 target_img_size=(512,512),
                 edge_loss_weight=1.0): # Pre konzistenciu, pri evaluácii sa nepoužije
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        if not self.image_list:
            print(f"VAROVANIE: Nenašli sa žiadne obrázky v {os.path.join(self.path, 'images')}")
        self.input_min_g, self.input_max_g = input_min_max_global
        if self.input_min_g is None or self.input_max_g is None:
            raise ValueError("input_min_max_global musí byť poskytnuté pre WrapCountDataset.")
        self.k_max = k_max_val
        self.target_img_size = target_img_size
        self.edge_loss_weight = edge_loss_weight

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data) if isinstance(data, torch.Tensor) else np.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _ensure_shape_and_type(self, img_numpy, target_shape, data_name="Image", dtype=np.float32):
        img_numpy = img_numpy.astype(dtype)
        current_shape = img_numpy.shape[-2:] # Funguje pre 2D aj 3D (C,H,W)
        
        if current_shape != target_shape:
            original_shape_for_debug = img_numpy.shape
            # Ak je menší, padneme
            h, w = current_shape
            target_h, target_w = target_shape
            
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                if img_numpy.ndim == 2: # (H, W)
                    img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: # (1, H, W)
                    img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                else: # Iné tvary (napr. RGB) nie sú očakávané pre tieto dáta
                    print(f"VAROVANIE: Neočakávaný tvar pre padding {data_name}: {img_numpy.shape}. Skúšam ako 2D.")
                    if img_numpy.ndim > 2: img_numpy = img_numpy.squeeze() # Skúsime odstrániť nadbytočné dimenzie
                    if img_numpy.ndim == 2:
                         img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                    else:
                         raise ValueError(f"Nepodporovaný tvar pre padding {data_name}: {original_shape_for_debug}")


            # Ak je väčší, orežeme (center crop)
            h, w = img_numpy.shape[-2:] # Znovu získame rozmery po prípadnom paddingu
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3 and img_numpy.shape[0] == 1: # (1, H, W)
                    img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]
                else:
                    print(f"VAROVANIE: Neočakávaný tvar pre cropping {data_name}: {img_numpy.shape}. Skúšam ako 2D.")
                    if img_numpy.ndim > 2: img_numpy = img_numpy.squeeze()
                    if img_numpy.ndim == 2:
                        img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                    else:
                        raise ValueError(f"Nepodporovaný tvar pre cropping {data_name}: {original_shape_for_debug}")


            if img_numpy.shape[-2:] != target_shape:
                 print(f"VAROVANIE: {data_name} '{getattr(self, 'current_img_path_for_debug', 'N/A')}' mal tvar {original_shape_for_debug}, po úprave na {target_shape} má {img_numpy.shape}. Môže dôjsť k chybe.")
        return img_numpy


    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        self.current_img_path_for_debug = self.image_list[index]
        img_path = self.image_list[index]
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')
        
        if not os.path.exists(img_path):
            print(f"CHYBA: Vstupný obrázok neexistuje: {img_path}"); return None, None, None, None, None
        if not os.path.exists(lbl_path):
            print(f"CHYBA: Label obrázok neexistuje: {lbl_path}"); return None, None, None, None, None

        try:
            wrapped_orig_np = tiff.imread(img_path)
            unwrapped_orig_np = tiff.imread(lbl_path)
        except Exception as e:
            print(f"CHYBA načítania TIFF: {img_path} alebo {lbl_path}. Error: {e}")
            return None, None, None, None, None

        wrapped_orig_np = self._ensure_shape_and_type(wrapped_orig_np, self.target_img_size, "Wrapped phase (static_eval)")
        unwrapped_orig_np = self._ensure_shape_and_type(unwrapped_orig_np, self.target_img_size, "Unwrapped phase (static_eval)")

        # Normalizovaný vstup pre model
        wrapped_input_norm_np = self._normalize_input_minmax_to_minus_one_one(wrapped_orig_np, self.input_min_g, self.input_max_g)
        wrapped_input_norm_tensor = torch.from_numpy(wrapped_input_norm_np.copy().astype(np.float32)).unsqueeze(0) # (1, H, W)

        # k-label pre metriku presnosti
        diff_np = (unwrapped_orig_np - wrapped_orig_np) / (2 * np.pi)
        k_float_np = np.round(diff_np)
        k_float_np = np.clip(k_float_np, -self.k_max, self.k_max)
        k_label_np = (k_float_np + self.k_max) #.astype(np.int64) # Pre cross_entropy by mal byť long
        k_label_tensor = torch.from_numpy(k_label_np.copy().astype(np.int64)) # (H,W)
        
        # Pôvodné dáta pre rekonštrukciu a MAE
        unwrapped_gt_orig_tensor = torch.from_numpy(unwrapped_orig_np.copy().astype(np.float32)) # (H,W)
        wrapped_orig_tensor = torch.from_numpy(wrapped_orig_np.copy().astype(np.float32))       # (H,W)
        
        # Weight map sa pri evaluácii zvyčajne nepoužíva, ale pre konzistenciu s collate_fn
        # môžeme vrátiť tensor jednotiek.
        weight_map_tensor = torch.ones_like(k_label_tensor, dtype=torch.float32)

        return wrapped_input_norm_tensor, k_label_tensor, unwrapped_gt_orig_tensor, wrapped_orig_tensor, weight_map_tensor

def collate_fn_skip_none_classification(batch): # Prevzaté z dwc.py
    # Filter out samples where any of the first 5 elements is None
    batch = list(filter(lambda x: all(item is not None for item in x[:5]), batch))
    if not batch: return None, None, None, None, None # Vráti 5 None hodnôt
    return torch.utils.data.dataloader.default_collate(batch)

def k_label_accuracy_full(logits, klabels): # Prevzaté z dwc.py
    # logits (B,C,H,W), klabels (B,H,W)
    pred_classes = torch.argmax(logits, dim=1) # (B,H,W)
    correct = (pred_classes == klabels).float().sum() # Počet správnych pixelov
    total = klabels.numel() # Celkový počet pixelov
    if total == 0: return torch.tensor(0.0) # Prípad prázdneho batchu
    return correct / total

# Metriky PSNR a SSIM (vyžaduje skimage) - zostáva z pôvodného
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Knižnica scikit-image nie je nainštalovaná. PSNR a SSIM nebudú vypočítané.")
    SKIMAGE_AVAILABLE = False

def calculate_psnr_ssim(gt_img_numpy, pred_img_numpy): # Zostáva z pôvodného
    if not SKIMAGE_AVAILABLE:
        return np.nan, np.nan
    
    gt_img_numpy = gt_img_numpy.squeeze()
    pred_img_numpy = pred_img_numpy.squeeze()

    data_range = gt_img_numpy.max() - gt_img_numpy.min()
    if data_range < 1e-6:
        current_psnr = float('inf') if np.allclose(gt_img_numpy, pred_img_numpy) else 0.0
    else:
        current_psnr = psnr(gt_img_numpy, pred_img_numpy, data_range=data_range)

    min_dim = min(gt_img_numpy.shape[-2:]) # Posledné dve dimenzie
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        current_ssim = np.nan
    else:
        try:
            current_ssim = ssim(gt_img_numpy, pred_img_numpy, data_range=data_range, channel_axis=None, win_size=win_size, gaussian_weights=True, use_sample_covariance=False)
        except ValueError: # Môže nastať, ak sú obrázky príliš malé alebo konštantné
            current_ssim = np.nan
            
    return current_psnr, current_ssim

def evaluate_and_visualize_model(
    config_path,
    weights_path,
    test_dataset_path,
    device_str='cuda'
    ):
    if not os.path.exists(config_path):
        print(f"CHYBA: Konfiguračný súbor nebol nájdený: {config_path}")
        return
    if not os.path.exists(weights_path):
        print(f"CHYBA: Súbor s váhami nebol nájdený: {weights_path}")
        return

    # --- Načítanie Konfigurácie ---
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
    
    print("--- Načítaná Konfigurácia Experimentu (Klasifikácia) ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    encoder_name = config.get("Encoder Name", "resnet34") 
    k_max_val_from_config = config.get("K_MAX")
    if k_max_val_from_config is None:
        print(f"VAROVANIE: 'K_MAX' nenájdené v configu, používam fallback: {KMAX_DEFAULT_FALLBACK}")
        k_max_val = KMAX_DEFAULT_FALLBACK
    else:
        k_max_val = int(k_max_val_from_config)
    
    num_classes_effective_from_config = config.get("NUM_CLASSES")
    if num_classes_effective_from_config is None:
        num_classes_effective = 2 * k_max_val + 1
        print(f"VAROVANIE: 'NUM_CLASSES' nenájdené v configu, vypočítavam z K_MAX: {num_classes_effective}")
    else:
        num_classes_effective = int(num_classes_effective_from_config)
        if num_classes_effective != (2 * k_max_val + 1):
            print(f"VAROVANIE: Nesúlad medzi NUM_CLASSES ({num_classes_effective}) a K_MAX ({k_max_val}) v configu.")
            
    input_norm_str = config.get("Input Normalization (Global MinMax for Wrapped)")
    global_input_min, global_input_max = None, None
    if input_norm_str:
        try:
            min_str, max_str = input_norm_str.split(',')
            global_input_min = float(min_str.split(':')[1].strip())
            global_input_max = float(max_str.split(':')[1].strip())
            print(f"Načítané globálne Min/Max pre vstup: Min={global_input_min:.4f}, Max={global_input_max:.4f}")
        except Exception as e:
            print(f"CHYBA pri parsovaní Input Normalization stats: {e}. Normalizácia vstupu nemusí byť správna.")
    else:
        print("CHYBA: 'Input Normalization (Global MinMax for Wrapped)' nenájdené v configu.")
        return

    if global_input_min is None or global_input_max is None:
        print("CHYBA: Nepodarilo sa načítať normalizačné štatistiky pre vstup. Končím.")
        return

    # --- Príprava Datasetu a DataLoaderu ---
    print(f"\nNačítavam testovací dataset (klasifikačný mód) z: {test_dataset_path}")
    test_dataset = WrapCountDataset(
        path_to_data=test_dataset_path,
        input_min_max_global=(global_input_min, global_input_max),
        k_max_val=k_max_val
    )
    if len(test_dataset) == 0:
        print("CHYBA: Testovací dataset je prázdny.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, 
                             collate_fn=collate_fn_skip_none_classification)
    
    # --- Načítanie Modelu ---
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    try:
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None, 
            in_channels=1,        
            classes=num_classes_effective, 
            activation=None
        ).to(device)
        print(f"Používam smp.Unet s enkóderom: {encoder_name}, Počet tried: {num_classes_effective}")
    except Exception as e:
        print(f"CHYBA pri inicializácii smp.Unet: {e}")
        return

    try:
        net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Váhy modelu úspešne načítané z: {weights_path}")
    except Exception as e:
        print(f"CHYBA pri načítaní váh modelu: {e}")
        return
    net.eval()

    # --- Evaluácia Celého Testovacieho Setu ---
    print("\nEvaluujem celý testovací dataset (klasifikačný model)...")
    all_mae_reconstructed = []
    all_k_accuracy = []
    all_psnr_reconstructed = []
    all_ssim_reconstructed = []
    
    all_pixel_errors_flat_rec = [] # Pre histogram chýb rekonštrukcie
    all_samples_data_for_avg_rec = [] # Pre nájdenie priemernej MAE vzorky

    # Informácie pre najlepší, najhorší a priemerný prípad MAE (rekonštrukcia)
    best_mae_sample_info_rec = {"mae": float('inf'), "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None}
    worst_mae_sample_info_rec = {"mae": -1.0, "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None}
    avg_mae_sample_info_rec = {"mae": -1.0, "index": -1, "wrapped_orig": None, "gt_unwrapped": None, "pred_unwrapped_reconstructed": None, "diff_from_avg_mae": float('inf')}
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader): # Použijeme tqdm tu
            if batch_data is None or batch_data[0] is None:
                print(f"Preskakujem chybný batch v teste, iterácia {i}")
                continue
            
            input_norm_batch, k_labels_gt_batch, unwrapped_gt_orig_batch, wrapped_orig_batch, _ = batch_data
            
            input_norm_batch = input_norm_batch.to(device)
            k_labels_gt_batch_dev = k_labels_gt_batch.to(device) 

            logits_batch = net(input_norm_batch) 
            pred_classes_batch = torch.argmax(logits_batch, dim=1) 

            current_k_acc = k_label_accuracy_full(logits_batch, k_labels_gt_batch_dev) 
            all_k_accuracy.append(current_k_acc.item())

            for k_idx in range(pred_classes_batch.size(0)):
                current_sample_global_idx = i * test_loader.batch_size + k_idx 
                
                pred_classes_sample = pred_classes_batch[k_idx].cpu() 
                wrapped_orig_sample_numpy = wrapped_orig_batch[k_idx].cpu().numpy().squeeze() 
                unwrapped_gt_orig_sample_numpy = unwrapped_gt_orig_batch[k_idx].cpu().numpy().squeeze() 

                k_pred_values_sample = pred_classes_sample.float() - k_max_val 
                unwrapped_pred_reconstructed_numpy = wrapped_orig_sample_numpy + (2 * np.pi) * k_pred_values_sample.numpy()

                # Zber chýb pre histogram
                current_pixel_errors_rec = np.abs(unwrapped_pred_reconstructed_numpy - unwrapped_gt_orig_sample_numpy)
                all_pixel_errors_flat_rec.extend(current_pixel_errors_rec.flatten().tolist())

                mae = np.mean(current_pixel_errors_rec)
                all_mae_reconstructed.append(mae)
                
                # Uloženie dát pre nájdenie priemernej MAE vzorky
                all_samples_data_for_avg_rec.append((mae, wrapped_orig_sample_numpy, unwrapped_gt_orig_sample_numpy, unwrapped_pred_reconstructed_numpy, current_sample_global_idx))


                if SKIMAGE_AVAILABLE:
                    psnr_val, ssim_val = calculate_psnr_ssim(unwrapped_gt_orig_sample_numpy, unwrapped_pred_reconstructed_numpy)
                    if not np.isnan(psnr_val): all_psnr_reconstructed.append(psnr_val)
                    if not np.isnan(ssim_val): all_ssim_reconstructed.append(ssim_val)
                
                if mae < best_mae_sample_info_rec["mae"]:
                    best_mae_sample_info_rec.update({"mae": mae, "index": current_sample_global_idx, 
                                                 "wrapped_orig": wrapped_orig_sample_numpy, 
                                                 "gt_unwrapped": unwrapped_gt_orig_sample_numpy, 
                                                 "pred_unwrapped_reconstructed": unwrapped_pred_reconstructed_numpy})
                
                if mae > worst_mae_sample_info_rec["mae"]:
                    worst_mae_sample_info_rec.update({"mae": mae, "index": current_sample_global_idx, 
                                                  "wrapped_orig": wrapped_orig_sample_numpy, 
                                                  "gt_unwrapped": unwrapped_gt_orig_sample_numpy, 
                                                  "pred_unwrapped_reconstructed": unwrapped_pred_reconstructed_numpy})

    avg_mae_rec = np.mean(all_mae_reconstructed) if all_mae_reconstructed else np.nan
    avg_k_acc = np.mean(all_k_accuracy) if all_k_accuracy else np.nan
    avg_psnr_rec = np.mean(all_psnr_reconstructed) if all_psnr_reconstructed else np.nan
    avg_ssim_rec = np.mean(all_ssim_reconstructed) if all_ssim_reconstructed else np.nan

    print("\n--- Celkové Priemerné Metriky na Testovacom Datasete (Klasifikácia & Rekonštrukcia) ---")
    print(f"Priemerná MAE (rekonštrukcia): {avg_mae_rec:.4f}")
    print(f"Priemerná k-label Accuracy: {avg_k_acc:.4f}")
    if SKIMAGE_AVAILABLE:
        print(f"Priemerný PSNR (rekonštrukcia): {avg_psnr_rec:.2f} dB")
        print(f"Priemerný SSIM (rekonštrukcia): {avg_ssim_rec:.4f}")
    
    # Nájdenie vzorky najbližšej k priemernej MAE (rekonštrukcia)
    if not np.isnan(avg_mae_rec) and all_samples_data_for_avg_rec:
        min_diff_to_avg_mae_rec = float('inf')
        avg_candidate_data_rec = None
        for sample_mae_val, s_wrapped, s_gt_unwrapped, s_pred_unwrapped, s_idx in all_samples_data_for_avg_rec:
            diff = abs(sample_mae_val - avg_mae_rec)
            if diff < min_diff_to_avg_mae_rec:
                min_diff_to_avg_mae_rec = diff
                avg_candidate_data_rec = (sample_mae_val, s_wrapped, s_gt_unwrapped, s_pred_unwrapped, s_idx)
        
        if avg_candidate_data_rec:
            avg_mae_sample_info_rec.update({
                "mae": avg_candidate_data_rec[0], 
                "wrapped_orig": avg_candidate_data_rec[1],
                "gt_unwrapped": avg_candidate_data_rec[2],
                "pred_unwrapped_reconstructed": avg_candidate_data_rec[3],
                "index": avg_candidate_data_rec[4],
                "diff_from_avg_mae": min_diff_to_avg_mae_rec
            })

    print("\n--- Extrémne a Priemerné Hodnoty MAE (Rekonštrukcia) ---")
    if best_mae_sample_info_rec["index"] != -1:
        print(f"Najlepšia MAE (rekon.): {best_mae_sample_info_rec['mae']:.4f} (index: {best_mae_sample_info_rec['index']})")
    if avg_mae_sample_info_rec["index"] != -1:
        print(f"Vzorka najbližšie k priemernej MAE (rekon. {avg_mae_rec:.4f}): MAE={avg_mae_sample_info_rec['mae']:.4f} (index: {avg_mae_sample_info_rec['index']}, rozdiel: {avg_mae_sample_info_rec['diff_from_avg_mae']:.4f})")
    if worst_mae_sample_info_rec["index"] != -1:
        print(f"Najhoršia MAE (rekon.): {worst_mae_sample_info_rec['mae']:.4f} (index: {worst_mae_sample_info_rec['index']})")

    run_name_for_file = os.path.splitext(os.path.basename(weights_path))[0].replace("best_weights_clf_", "eval_clf_")

    # Uloženie extrémnych a priemerných MAE hodnôt do textového súboru
    if best_mae_sample_info_rec["index"] != -1 or worst_mae_sample_info_rec["index"] != -1 or avg_mae_sample_info_rec["index"] != -1:
        extreme_mae_log_path_rec = f"extreme_mae_values_reconstruction_{run_name_for_file}.txt"
        with open(extreme_mae_log_path_rec, 'w') as f:
            f.write(f"Experiment (Klasifikácia & Rekonštrukcia): {run_name_for_file}\n")
            f.write("--- Extrémne a Priemerné Hodnoty MAE (Rekonštrukcia) ---\n")
            if best_mae_sample_info_rec["index"] != -1:
                f.write(f"Najlepšia MAE (rekon.): {best_mae_sample_info_rec['mae']:.6f} (index: {best_mae_sample_info_rec['index']})\n")
            if avg_mae_sample_info_rec["index"] != -1:
                f.write(f"Vzorka najbližšie k priemernej MAE (rekon. {avg_mae_rec:.6f}): MAE={avg_mae_sample_info_rec['mae']:.6f} (index: {avg_mae_sample_info_rec['index']}, rozdiel: {avg_mae_sample_info_rec['diff_from_avg_mae']:.6f})\n")
            if worst_mae_sample_info_rec["index"] != -1:
                f.write(f"Najhoršia MAE (rekon.): {worst_mae_sample_info_rec['mae']:.6f} (index: {worst_mae_sample_info_rec['index']})\n")
            f.write(f"\nPriemerná MAE (rekonštrukcia, celý dataset): {avg_mae_rec:.6f}\n")
            f.write(f"Priemerná k-label Accuracy (celý dataset): {avg_k_acc:.6f}\n")
        print(f"Extrémne a priemerné MAE (rekon.) hodnoty uložené do: {extreme_mae_log_path_rec}")

    # --- Histogram Chýb Rekonštrukcie ---
    if all_pixel_errors_flat_rec:
        all_pixel_errors_flat_rec_np = np.array(all_pixel_errors_flat_rec)
        plt.figure(figsize=(12, 7))
        plt.hist(all_pixel_errors_flat_rec_np, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.title('Histogram Absolútnych Chýb Rekonštrukcie (všetky pixely)', fontsize=16)
        plt.xlabel('Absolútna Chyba Rekonštrukcie (radiány)', fontsize=14)
        plt.ylabel('Počet Pixelov', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.yscale('log') 
        hist_rec_save_path_png = f"error_histogram_reconstruction_{run_name_for_file}.png"
        hist_rec_save_path_svg = f"error_histogram_reconstruction_{run_name_for_file}.svg"
        plt.savefig(hist_rec_save_path_png)
        plt.savefig(hist_rec_save_path_svg)
        print(f"Histogram chýb rekonštrukcie uložený do: {hist_rec_save_path_png} a {hist_rec_save_path_svg}")
        plt.show()
        plt.close()

    # --- Vizualizácia Najlepšej, Priemernej a Najhoršej MAE (Rekonštrukcia) s Mapami Chýb ---
    if best_mae_sample_info_rec["index"] != -1 and worst_mae_sample_info_rec["index"] != -1 and avg_mae_sample_info_rec["index"] != -1:
        print(f"\nVizualizujem a ukladám najlepší, priemerný a najhorší MAE prípad (rekonštrukcia)...")
        
        samples_to_plot_rec = [
            ("Min MAE (Rekon.)", best_mae_sample_info_rec),
            ("Avg MAE (Rekon.)", avg_mae_sample_info_rec),
            ("Max MAE (Rekon.)", worst_mae_sample_info_rec)
        ]

        all_wrapped_col1 = []
        all_gt_pred_unwrapped_col23 = []

        for _, sample_info in samples_to_plot_rec:
            all_wrapped_col1.append(sample_info["wrapped_orig"])
            all_gt_pred_unwrapped_col23.append(sample_info["gt_unwrapped"])
            all_gt_pred_unwrapped_col23.append(sample_info["pred_unwrapped_reconstructed"])

        # Globálne min/max pre konzistentné škály v stĺpcoch
        vmin_col1_rec = np.min([img.min() for img in all_wrapped_col1 if img is not None]) if any(img is not None for img in all_wrapped_col1) else 0
        vmax_col1_rec = np.max([img.max() for img in all_wrapped_col1 if img is not None]) if any(img is not None for img in all_wrapped_col1) else 1
        if vmax_col1_rec <= vmin_col1_rec: vmax_col1_rec = vmin_col1_rec + 1e-5


        vmin_col23_rec = np.min([img.min() for img in all_gt_pred_unwrapped_col23 if img is not None]) if any(img is not None for img in all_gt_pred_unwrapped_col23) else 0
        vmax_col23_rec = np.max([img.max() for img in all_gt_pred_unwrapped_col23 if img is not None]) if any(img is not None for img in all_gt_pred_unwrapped_col23) else 1
        if vmax_col23_rec <= vmin_col23_rec: vmax_col23_rec = vmin_col23_rec + 1e-5
        
        fig, axs = plt.subplots(3, 4, figsize=(16, 13)) # Zmenené figsize

        # Názvy stĺpcov podľa vysledky_drg.py
        col_titles_aligned = ["Zabalený obraz", "Rozbalený referenčný obraz", "Predikcia", "Absolútna chyba"]
        # row_titles_rec sú v sample_info[0] a nebudú sa nastavovať ako ylabel

        error_map_mappables_rec = []
        img0_for_cbar, img1_for_cbar = None, None # Pre zdieľané colorbary

        for i, (row_desc, sample_info) in enumerate(samples_to_plot_rec):
            # Stĺpec 1: Zabalený obraz
            current_img0 = axs[i, 0].imshow(sample_info["wrapped_orig"], cmap='gray', vmin=vmin_col1_rec, vmax=vmax_col1_rec)
            if i == 0: img0_for_cbar = current_img0 # Uložíme pre zdieľaný colorbar

            # Stĺpec 2: Rozbalený referenčný
            current_img1 = axs[i, 1].imshow(sample_info["gt_unwrapped"], cmap='gray', vmin=vmin_col23_rec, vmax=vmax_col23_rec)
            if i == 0: img1_for_cbar = current_img1 # Uložíme pre zdieľaný colorbar
            
            # Stĺpec 3: Rekonštrukcia (Predikcia)
            axs[i, 2].imshow(sample_info["pred_unwrapped_reconstructed"], cmap='gray', vmin=vmin_col23_rec, vmax=vmax_col23_rec)
            
            # Stĺpec 4: Absolútna chyba
            error_map_rec = np.abs(sample_info["pred_unwrapped_reconstructed"] - sample_info["gt_unwrapped"])
            # Zabezpečenie, že vmax je vždy väčšie ako vmin pre individuálnu škálu chyby
            err_min_val = error_map_rec.min()
            err_max_val = error_map_rec.max()
            if err_max_val <= err_min_val:
                err_max_val = err_min_val + 1e-5
            
            img3 = axs[i, 3].imshow(error_map_rec, cmap='viridis', vmin=err_min_val, vmax=err_max_val) # Individuálne škály pre chyby
            error_map_mappables_rec.append(img3)

            # Odstránené axs[i,0].set_ylabel(...)

        for j, col_title_text in enumerate(col_titles_aligned):
            axs[0, j].set_title(col_title_text, fontsize=16, pad=20)

        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        
        # Individuálne farebné škály pre mapy chýb (ako v drg)
        for i in range(3):
            fig.colorbar(error_map_mappables_rec[i], ax=axs[i, 3], orientation='vertical', fraction=0.046, pad=0.02, aspect=15)

        # Úprava rozloženia podľa vysledky_drg.py
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.90, wspace=0.0, hspace=0.06)

        # Spoločný colorbar pre 1. stĺpec (ako v drg)
        if img0_for_cbar:
            pos_col0_ax2 = axs[2,0].get_position()
            cax1_left = pos_col0_ax2.x0
            cax1_bottom = 0.13 
            cax1_width = pos_col0_ax2.width
            cax1_height = 0.025 
            cax1 = fig.add_axes([cax1_left, cax1_bottom, cax1_width, cax1_height])
            cb1 = fig.colorbar(img0_for_cbar, cax=cax1, orientation='horizontal')
            # cb1.set_ticks(...) # Ponecháme default ticks ako v drg

        # Spoločný colorbar pre 2. a 3. stĺpec (ako v drg)
        if img1_for_cbar:
            pos_col1_ax2 = axs[2,1].get_position() 
            pos_col2_ax2 = axs[2,2].get_position() 
            cax23_left = pos_col1_ax2.x0
            cax23_bottom = 0.13 
            cax23_width = (pos_col2_ax2.x0 + pos_col2_ax2.width) - pos_col1_ax2.x0 
            cax23_height = 0.025
            cax23 = fig.add_axes([cax23_left, cax23_bottom, cax23_width, cax23_height])
            cb23 = fig.colorbar(img1_for_cbar, cax=cax23, orientation='horizontal') 
            # cb23.set_ticks(...) # Ponecháme default ticks ako v drg
        
        base_save_name_rec = f"detailed_comparison_mae_reconstruction_{run_name_for_file}"
        save_fig_path_png_rec = f"{base_save_name_rec}.png"
        save_fig_path_svg_rec = f"{base_save_name_rec}.svg"

        plt.savefig(save_fig_path_png_rec, dpi=200, bbox_inches='tight') 
        print(f"Detailná vizualizácia (rekonštrukcia) uložená do: {save_fig_path_png_rec}")
        plt.savefig(save_fig_path_svg_rec, bbox_inches='tight')
        print(f"Detailná vizualizácia (rekonštrukcia) uložená aj do: {save_fig_path_svg_rec}")
        plt.show()
        plt.close(fig) # Zatvoríme figúru explicitne
    else:
        print("Nepodarilo sa nájsť dostatok dát pre plnú detailnú vizualizáciu (rekonštrukcia).")


if __name__ == '__main__':
    # --- NASTAVENIA PRE TESTOVANIE (KLASIFIKÁCIA) ---
    # Tieto cesty musia smerovať na výstupy z klasifikačného tréningu (dwc.py)
    # CONFIG_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment3_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW3.0_bs8.txt" # PRÍKLAD! UPRAV!
    # WEIGHTS_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment3_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW3.0_bs8.pth" # PRÍKLAD! UPRAV!
    
    # TEST_DATA_PATH = r'C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset' # Zostáva rovnaký
    


    # CONFIG_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment5_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW4.0_bs8.txt" # PRÍKLAD! UPRAV!
    # WEIGHTS_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment5_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW4.0_bs8.pth" # PRÍKLAD! UPRAV!
    
    # TEST_DATA_PATH = r'C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset' # Zostáva rovnaký

    CONFIG_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\config_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.txt" # PRÍKLAD! UPRAV!
    WEIGHTS_FILE_PATH = r"C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\classification\experiment4_hyper\best_weights_clf_R34imgnet_Kmax6_AugMed_LR1em03_WD1em04_Ep120_Tmax120_EtaMin1em07_EdgeW5.0_bs8.pth" # PRÍKLAD! UPRAV!
    
    TEST_DATA_PATH = r'C:\Users\juraj\Desktop\TRENOVANIE_bakalarka_simul\split_dataset_tiff_for_dynamic_v_stratified_final\static_test_dataset' # Zostáva rovnaký


    DEVICE_TO_USE = 'cuda'

    script_start_time = time.time()

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"CHYBA: Konfiguračný súbor '{CONFIG_FILE_PATH}' neexistuje. Skontroluj cestu.")
    elif not os.path.exists(WEIGHTS_FILE_PATH):
        print(f"CHYBA: Súbor s váhami '{WEIGHTS_FILE_PATH}' neexistuje. Skontroluj cestu.")
    else:
        evaluate_and_visualize_model(
            config_path=CONFIG_FILE_PATH,
            weights_path=WEIGHTS_FILE_PATH,
            test_dataset_path=TEST_DATA_PATH,
            device_str=DEVICE_TO_USE
        )
    
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    print(f"\nCelkový čas vykonávania skriptu: {total_script_time:.2f} sekúnd ({time.strftime('%H:%M:%S', time.gmtime(total_script_time))}).")