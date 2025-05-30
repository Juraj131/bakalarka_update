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
from skimage import io             # <--- PRIDANÉ
from skimage.filters import gaussian # <--- PRIDANÉ

# --- Globálne Konštanty pre Augmentácie (Normalizovaný Vstup [-1,1]) ---
# Tieto zostávajú, ak ich používate v AddGaussianNoiseTransform alebo inde
NORMALIZED_INPUT_CLAMP_MIN = -1.0
NORMALIZED_INPUT_CLAMP_MAX = 1.0
NORMALIZED_INPUT_ERASING_VALUE = 0.0 

# --- Sem patria vaše funkcie z pôvodného simulačného skriptu ---
# get_random_or_fixed, wrap_phase, generate_cubic_background
# a upravená generate_simulation_pair_from_source_np
# (Skopírujem ich sem pre úplnosť, aby bol tento skript samostatne funkčný)

def get_random_or_fixed(param_value, is_integer=False, allow_float_for_int=False):
    if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
        min_val, max_val = param_value
        if min_val > max_val:
            min_val_temp = min_val
            min_val = max_val
            max_val = min_val_temp
        if is_integer:
            low = int(round(min_val))
            high = int(round(max_val))
            if high < low: high = low
            if not allow_float_for_int and isinstance(min_val, int) and isinstance(max_val, int):
                return np.random.randint(min_val, max_val + 1)
            else:
                return int(np.random.randint(low, high + 1))
        else:
            return np.random.uniform(min_val, max_val)
    return param_value

def wrap_phase(img):
    return (img + np.pi) % (2 * np.pi) - np.pi

def generate_cubic_background(shape, coeff_stats, scale=1.0, amplify_ab=1.0, n_strips=6, tilt_angle_deg=0.0):
    H, W = shape
    y_idxs, x_idxs = np.indices((H, W))
    slope_y = (n_strips * 2 * np.pi) / H
    tilt_angle_rad = np.deg2rad(tilt_angle_deg)
    slope_x = slope_y * np.tan(tilt_angle_rad)
    linear_grad = slope_y * y_idxs + slope_x * x_idxs
    x_norm = np.linspace(0, 1, W)
    a_val = np.random.normal(coeff_stats[0][0], coeff_stats[0][1] * scale)
    b_val = np.random.normal(coeff_stats[1][0], coeff_stats[1][1] * scale)
    c_val = np.random.normal(coeff_stats[2][0], coeff_stats[2][1] * scale)
    d_val = np.random.normal(coeff_stats[3][0], coeff_stats[3][1] * scale)
    a_val *= amplify_ab
    b_val *= amplify_ab
    poly1d = a_val * x_norm**3 + b_val * x_norm**2 + c_val * x_norm + d_val
    poly2d = np.tile(poly1d, (H, 1))
    background = linear_grad + poly2d
    return background, dict(a=a_val, b=b_val, c=c_val, d=d_val, n_strips_actual=n_strips,
                            slope_y=slope_y, slope_x=slope_x, tilt_angle_deg=tilt_angle_deg)

def generate_simulation_pair_from_source_np_for_training( # Mierne iný názov pre jasnosť
    source_image_np, # Vstup je už normalizovaný fluorescenčný obraz [0,1]
    param_ranges_config, 
    amplify_ab_value
    # verbose_simulation nie je potrebné pre dynamické generovanie počas tréningu
):
    n_strips = get_random_or_fixed(param_ranges_config["n_strips_param"], is_integer=True, allow_float_for_int=True)
    original_image_influence = get_random_or_fixed(param_ranges_config["original_image_influence_param"])
    phase_noise_std = get_random_or_fixed(param_ranges_config["phase_noise_std_param"])
    smooth_original_image_sigma = get_random_or_fixed(param_ranges_config["smooth_original_image_sigma_param"])
    poly_scale = get_random_or_fixed(param_ranges_config["poly_scale_param"])
    CURVATURE_AMPLITUDE = get_random_or_fixed(param_ranges_config["CURVATURE_AMPLITUDE_param"])
    background_d_offset = get_random_or_fixed(param_ranges_config["background_offset_d_param"])
    tilt_angle_deg = get_random_or_fixed(param_ranges_config["tilt_angle_deg_param"])

    coeff_stats_for_bg_generation = [
        (0.0, 0.3 * CURVATURE_AMPLITUDE),
        (-4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (+4.0 * CURVATURE_AMPLITUDE, 0.3 * CURVATURE_AMPLITUDE),
        (background_d_offset, 2.0) 
    ]

    if smooth_original_image_sigma > 0 and original_image_influence > 0: # Aplikuje sa na source_image_np
        img_phase_obj_base = gaussian(source_image_np, sigma=smooth_original_image_sigma, preserve_range=True, channel_axis=None)
    else:
        img_phase_obj_base = source_image_np
    object_phase_contribution = img_phase_obj_base * (2 * np.pi)

    generated_background, _ = generate_cubic_background(
        source_image_np.shape, coeff_stats_for_bg_generation, 
        scale=poly_scale, amplify_ab=amplify_ab_value, 
        n_strips=n_strips, tilt_angle_deg=tilt_angle_deg
    )
    
    unwrapped_phase = (object_phase_contribution * original_image_influence) + \
                      (generated_background * (1.0 - original_image_influence))

    if phase_noise_std > 0:
        unwrapped_phase += np.random.normal(0, phase_noise_std, size=source_image_np.shape)
    wrapped_phase = wrap_phase(unwrapped_phase)
    return unwrapped_phase.astype(np.float32), wrapped_phase.astype(np.float32)

# --- Koniec kópií funkcií zo simulačného skriptu ---


# --- Funkcie na výpočet Štatistík (upravené pre použitie so statickým ref. tréningovým setom) ---
def get_global_input_min_max_stats(dataset_path, data_type_name="Input Data"):
    file_list = glob.glob(os.path.join(dataset_path, 'images', "*.tiff"))
    if not file_list: raise ValueError(f"Prázdny zoznam súborov pre {data_type_name} v {dataset_path}.")
    # ... (zvyšok funkcie get_input_min_max_stats zostáva rovnaký) ...
    min_val_g, max_val_g = np.inf, -np.inf
    print(f"Počítam Globálne Min/Max pre {data_type_name} z {len(file_list)} súborov v {dataset_path}...")
    for i, fp in enumerate(file_list):
        try:
            img = tiff.imread(fp).astype(np.float32)
            min_val_g, max_val_g = min(min_val_g, img.min()), max(max_val_g, img.max())
            if (i+1)%100==0 or (i+1)==len(file_list): print(f"  Spracovaných {i+1}/{len(file_list)}...")
        except Exception as e: print(f"Chyba pri {fp}: {e}")
    if np.isinf(min_val_g) or np.isinf(max_val_g): raise ValueError(f"Nepodarilo sa načítať Min/Max pre {data_type_name}.")
    print(f"Výpočet Globálnych Min/Max pre {data_type_name} dokončený.")
    return min_val_g, max_val_g


def get_global_target_mean_std_stats(dataset_path, data_type_name="Target Data"):
    file_list = glob.glob(os.path.join(dataset_path, 'labels', "*.tiff"))
    if not file_list: raise ValueError(f"Prázdny zoznam súborov pre {data_type_name} v {dataset_path}.")
    # ... (zvyšok funkcie get_target_mean_std_stats zostáva rovnaký) ...
    all_vals_for_stats = []
    print(f"Počítam Globálne Priemer/Std pre {data_type_name} z {len(file_list)} súborov v {dataset_path}...")
    for i, fp in enumerate(file_list):
        try:
            img = tiff.imread(fp).astype(np.float32)
            all_vals_for_stats.append(img.flatten())
            if (i+1)%100==0 or (i+1)==len(file_list): print(f"  Spracovaných {i+1}/{len(file_list)}...")
        except Exception as e: print(f"Chyba pri {fp}: {e}")
    if not all_vals_for_stats: raise ValueError(f"Nepodarilo sa načítať dáta pre Priemer/Std pre {data_type_name}.")
    cat_vals = np.concatenate(all_vals_for_stats)
    mean_g, std_g = np.mean(cat_vals), np.std(cat_vals)
    print(f"Výpočet Globálnych Priemer/Std pre {data_type_name} dokončený.")
    return mean_g, std_g


# --- Augmentačné transformácie (zostávajú rovnaké) ---
class AddGaussianNoiseTransform(nn.Module):
    # ... (kód triedy zostáva rovnaký) ...
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

# --- Statický Dataset (pre validáciu a testovanie, prípadne pre referenčný tréningový) ---
class StaticPhaseDataset(Dataset): # Premenované z CustomDataset
    def __init__(self, path_to_data, 
                 input_processing_type, 
                 norm_stats_input_global, # Globálne štatistiky
                 norm_stats_target_global, # Globálne štatistiky
                 # Augmentácia sa tu nebude aplikovať, len pre tréningový dynamický
                 target_img_size=(512,512)): 
        self.path = path_to_data
        self.image_list = sorted(glob.glob(os.path.join(self.path, 'images', "*.tiff")))
        
        self.input_processing_type = input_processing_type
        self.input_min_g, self.input_max_g = (None, None)
        if input_processing_type == 'direct_minmax':
            if norm_stats_input_global is None:
                raise ValueError("norm_stats_input_global (min,max) musí byť poskytnutý pre 'direct_minmax'.")
            self.input_min_g, self.input_max_g = norm_stats_input_global

        self.target_mean_g, self.target_std_g = norm_stats_target_global
        self.target_img_size = target_img_size

    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0

    def _normalize_target_z_score(self, data, mean_val, std_val):
        if std_val < 1e-6: return data - mean_val 
        return (data - mean_val) / std_val

    def _ensure_shape_and_type(self, img_numpy, target_shape, dtype=np.float32):
        # ... (táto funkcia zostáva rovnaká, len bez data_name a debug atribútu) ...
        img_numpy = img_numpy.astype(dtype) 
        if img_numpy.shape[-2:] != target_shape:
            h, w = img_numpy.shape[-2:]
            target_h, target_w = target_shape
            pad_h = max(0, target_h - h); pad_w = max(0, target_w - w)
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2; pad_right = pad_w - pad_left
                if img_numpy.ndim == 2:
                    img_numpy = np.pad(img_numpy, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                elif img_numpy.ndim == 3:
                    img_numpy = np.pad(img_numpy, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            h, w = img_numpy.shape[-2:]
            if h > target_h or w > target_w:
                start_h = (h - target_h) // 2; start_w = (w - target_w) // 2
                if img_numpy.ndim == 2:
                    img_numpy = img_numpy[start_h:start_h+target_h, start_w:start_w+target_w]
                elif img_numpy.ndim == 3:
                    img_numpy = img_numpy[:, start_h:start_h+target_h, start_w:start_w+target_w]
            if img_numpy.shape[-2:] != target_shape:
                  raise ValueError(f"Ensure shape failed. Got {img_numpy.shape}, expected H,W as {target_shape}")
        return img_numpy


    def __len__(self): return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        # Odvodenie cesty k labelu na základe konvencie z vášho split skriptu
        base_id_name = os.path.basename(img_path).replace('wrappedbg_', '').replace('.tiff','')
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', f'unwrapped_{base_id_name}.tiff')

        try:
            wrapped_orig_phase = tiff.imread(img_path) 
            unwrapped_orig = tiff.imread(lbl_path)
        except Exception as e: 
            print(f"CHYBA načítania statického páru: {img_path} alebo {lbl_path}. Error: {e}"); return None,None
        
        wrapped_orig_phase = self._ensure_shape_and_type(wrapped_orig_phase, self.target_img_size)
        unwrapped_orig = self._ensure_shape_and_type(unwrapped_orig, self.target_img_size)

        if self.input_processing_type == 'sincos':
            # Pre statické dáta by ste museli mať uložené sin/cos alebo ich tu generovať
            # Ak sú statické dáta vždy 'direct_minmax', táto vetva sa nepoužije pre ne.
            # Pre jednoduchosť predpokladám, že statické dáta budú 'direct_minmax'
            raise NotImplementedError("SinCos pre statický dataset nie je plne implementovaný v tomto príklade.")
        elif self.input_processing_type == 'direct_minmax':
            wrapped_tensor_orig = torch.from_numpy(wrapped_orig_phase.copy().astype(np.float32))
            wrapped_input_tensor = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor_orig, self.input_min_g, self.input_max_g).unsqueeze(0)
        else: 
            raise ValueError(f"Neznámy input_processing_type: {self.input_processing_type}")
        
        unwrapped_tensor_orig = torch.from_numpy(unwrapped_orig.copy().astype(np.float32))
        unwrapped_target_tensor = self._normalize_target_z_score(unwrapped_tensor_orig, self.target_mean_g, self.target_std_g).unsqueeze(0) 
        
        return wrapped_input_tensor, unwrapped_target_tensor

# --- Dynamický Dataset pre Tréning ---
class DynamicTrainPhaseDataset(Dataset):
    def __init__(self, 
                 source_image_filepaths, # Zoznam ciest k zdrojovým fluorescenčným obrázkom
                 simulation_param_ranges, # Slovník s rozsahmi pre simuláciu
                 amplify_ab_fixed_value,  # Fixná hodnota pre amplify_ab
                 input_processing_type,   # 'direct_minmax' alebo 'sincos'
                 norm_stats_input_global, # Globálne (min_g, max_g) pre vstup
                 norm_stats_target_global,# Globálne (mean_g, std_g) pre cieľ
                 augmentation_strength='none', 
                 target_img_size=(512,512)):
        
        self.source_image_filepaths = source_image_filepaths
        self.simulation_param_ranges = simulation_param_ranges
        self.amplify_ab_fixed_value = amplify_ab_fixed_value
        
        self.input_processing_type = input_processing_type
        self.input_min_g, self.input_max_g = norm_stats_input_global
        self.target_mean_g, self.target_std_g = norm_stats_target_global
        
        self.augmentation_strength = augmentation_strength
        self.target_img_size = target_img_size # Zdrojové obrázky by už mali mať túto veľkosť

        self.geometric_transforms = None
        self.pixel_transforms = None
        if self.augmentation_strength != 'none':
            self._setup_augmentations(self.augmentation_strength)

    def _setup_augmentations(self, strength):
        # ... (kód pre _setup_augmentations z vášho pôvodného tréningového skriptu) ...
        # (T.RandomHorizontalFlip, AddGaussianNoiseTransform, T.RandomErasing)
        noise_std_range, noise_p = (0.01, 0.05), 0.0
        erase_scale, erase_p = (0.01, 0.04), 0.0
        if strength == 'light':
            noise_std_range, noise_p = (0.02, 0.08), 0.4
            erase_scale, erase_p = (0.01, 0.05), 0.3
        elif strength == 'medium':
            noise_std_range, noise_p = (0.03, 0.12), 0.5
            erase_scale, erase_p = (0.02, 0.08), 0.4
        elif strength == 'strong':
            noise_std_range, noise_p = (0.05, 0.15), 0.6
            erase_scale, erase_p = (0.02, 0.10), 0.5
        
        self.geometric_transforms = T.Compose([T.RandomHorizontalFlip(p=0.5)])
        
        pixel_aug_list = []
        if noise_p > 0:
            pixel_aug_list.append(AddGaussianNoiseTransform(std_dev_range=noise_std_range, p=noise_p,
                                        clamp_min=NORMALIZED_INPUT_CLAMP_MIN,
                                        clamp_max=NORMALIZED_INPUT_CLAMP_MAX))
        if erase_p > 0:
            erase_value = NORMALIZED_INPUT_ERASING_VALUE
            if self.input_processing_type == 'sincos': # sin/cos potrebuje 2 hodnoty pre mazanie
                erase_value = [NORMALIZED_INPUT_ERASING_VALUE, NORMALIZED_INPUT_ERASING_VALUE]
            pixel_aug_list.append(T.RandomErasing(p=erase_p, scale=erase_scale, ratio=(0.3, 3.3), 
                                        value=erase_value, inplace=False))
        if pixel_aug_list:
            self.pixel_transforms = T.Compose(pixel_aug_list)


    def _normalize_input_minmax_to_minus_one_one(self, data, min_val, max_val):
        # ... (rovnaká ako v StaticPhaseDataset) ...
        if max_val == min_val: return torch.zeros_like(data)
        return 2.0 * (data - min_val) / (max_val - min_val) - 1.0


    def _normalize_target_z_score(self, data, mean_val, std_val):
        # ... (rovnaká ako v StaticPhaseDataset) ...
        if std_val < 1e-6: return data - mean_val 
        return (data - mean_val) / std_val

    def __len__(self):
        return len(self.source_image_filepaths)

    def __getitem__(self, idx):
        source_img_path = self.source_image_filepaths[idx]
        try:
            source_img_raw = io.imread(source_img_path).astype(np.float32)
        except Exception as e:
            print(f"CHYBA načítania zdrojového obrázka pre dynamický tréning: {source_img_path}. Error: {e}"); 
            # Vrátime None, aby collate_fn mohol preskočiť
            return None, None 

        # Normalizácia zdrojového fluorescenčného obrazu na [0,1]
        img_min_val, img_max_val = source_img_raw.min(), source_img_raw.max()
        source_img_norm = (source_img_raw - img_min_val) / (img_max_val - img_min_val) if img_max_val > img_min_val else np.zeros_like(source_img_raw)
        
        # Dynamické generovanie syntetického páru
        unwrapped_phase_np, wrapped_phase_np = generate_simulation_pair_from_source_np_for_training(
            source_img_norm,
            self.simulation_param_ranges,
            self.amplify_ab_fixed_value
        )

        # Normalizácia a konverzia na tenzory
        if self.input_processing_type == 'direct_minmax':
            wrapped_tensor_orig = torch.from_numpy(wrapped_phase_np.copy()) # Už je float32
            wrapped_input_tensor = self._normalize_input_minmax_to_minus_one_one(wrapped_tensor_orig, self.input_min_g, self.input_max_g).unsqueeze(0)
        elif self.input_processing_type == 'sincos':
            sin_phi = torch.sin(torch.from_numpy(wrapped_phase_np.copy()))
            cos_phi = torch.cos(torch.from_numpy(wrapped_phase_np.copy()))
            wrapped_input_tensor = torch.stack([sin_phi, cos_phi], dim=0) # (2, H, W)
        else:
            raise ValueError(f"Neznámy input_processing_type: {self.input_processing_type}")
            
        unwrapped_tensor_orig = torch.from_numpy(unwrapped_phase_np.copy()) # Už je float32
        unwrapped_target_tensor = self._normalize_target_z_score(unwrapped_tensor_orig, self.target_mean_g, self.target_std_g).unsqueeze(0)
        
        # Aplikácia online augmentácií (po normalizácii, ak je to tak zamýšľané)
        # Geometrické augmentácie sa aplikujú na wrapped aj unwrapped
        if self.geometric_transforms:
            # torchvision.transforms.v2 by mal zvládnuť tuple vstupov
            wrapped_input_tensor, unwrapped_target_tensor = self.geometric_transforms(wrapped_input_tensor, unwrapped_target_tensor)
        
        # Pixelové augmentácie sa zvyčajne aplikujú len na vstup (wrapped)
        if self.pixel_transforms:
            wrapped_input_tensor = self.pixel_transforms(wrapped_input_tensor)
            
        return wrapped_input_tensor, unwrapped_target_tensor

# --- Ostatné pomocné funkcie (collate_fn, losses, denormalize) zostávajú rovnaké ---
def collate_fn_skip_none(batch):
    # ... (kód zostáva rovnaký) ...
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch: return None, None 
    return torch.utils.data.dataloader.default_collate(batch)

def mae_loss_on_normalized(p, t): return torch.mean(torch.abs(p-t))
def pixel_mse_loss(p, t): return F.mse_loss(p,t)
def sobel_gradient_loss(yt,yp,d):
    # ... (kód zostáva rovnaký) ...
    if yt.ndim==3: yt=yt.unsqueeze(1)
    if yp.ndim==3: yp=yp.unsqueeze(1)
    sx_w = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=d).view(1,1,3,3)
    sy_w = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=d).view(1,1,3,3)
    gx_t=F.conv2d(yt,sx_w,padding=1); gy_t=F.conv2d(yt,sy_w,padding=1)
    gx_p=F.conv2d(yp,sx_w,padding=1); gy_p=F.conv2d(yp,sy_w,padding=1)
    return torch.mean(torch.abs(gx_t-gx_p)) + torch.mean(torch.abs(gy_t-gy_p))

def denormalize_target_z_score(dn, om, os): return dn*os+om if os>1e-6 else torch.full_like(dn,om)

def denormalize_input_minmax_from_minus_one_one(data_norm_minus_one_one, original_min, original_max):
    """Denormalizuje dáta z rozsahu [-1, 1] na pôvodný rozsah [original_min, original_max]."""
    if original_max == original_min:
        return torch.full_like(data_norm_minus_one_one, original_min)
    return (data_norm_minus_one_one + 1.0) * (original_max - original_min) / 2.0 + original_min

# --- Tréningová funkcia (zostáva v princípe rovnaká, len s novými DataLoader-mi) ---
def run_training_session(
    run_id, device, num_epochs, 
    train_loader, # Bude pre DynamicTrainPhaseDataset
    val_loader,   # Bude pre StaticPhaseDataset
    test_loader,  # Bude pre StaticPhaseDataset
    target_original_mean, target_original_std, # Tieto sú teraz GLOBAL_...
    input_original_min_max, # Toto je teraz GLOBAL_...
    encoder_name, encoder_weights, input_processing_type, loss_type, lambda_gdl,
    learning_rate, weight_decay, 
    cosine_T_max, cosine_eta_min, 
    min_lr, 
    early_stopping_patience, augmentation_strength_train # Zmenené pre tréningový dataset
    ):
    # ... (obsah funkcie run_training_session zostáva rovnaký ako vo vašom kóde) ...
    config_save_path = f'config_{run_id}.txt'
    config_details = {
        "Run ID": run_id, "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Encoder Name": encoder_name, "Encoder Weights": encoder_weights,
        "Input Processing": input_processing_type, 
        "Train Augmentation Strength": augmentation_strength_train, 
        "Train Data Source": "Dynamic Simulation", 
        "Target Norm (Z-score) Mean": f"{target_original_mean:.4f}", 
        "Target Norm (Z-score) Std": f"{target_original_std:.4f}",
        "Loss Type": loss_type, "Lambda GDL": lambda_gdl if 'gdl' in loss_type else "N/A",
        "Initial LR": learning_rate, "Batch Size": train_loader.batch_size, 
        "Num Epochs": num_epochs, 
        "Weight Decay": weight_decay, # Pridané weight_decay do configu
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
        
        # Dôležité: Tu sa v každej iterácii (pre každý batch) budú dynamicky generovať dáta
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data[0] is None or batch_data[1] is None : # Kontrola pre collate_fn_skip_none
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Preskakujem None batch.")
                continue
            
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

        # Validácia (na statických dátach)
        net.eval()
        epoch_val_loss_n, epoch_val_mae_d = [], []
        with torch.no_grad():
            for batch_data_val in val_loader:
                if batch_data_val[0] is None or batch_data_val[1] is None: continue
                inputs_n_val, targets_n_val = batch_data_val[0].to(device), batch_data_val[1].to(device)
                preds_n_val = net(inputs_n_val)
                main_loss_v = mae_loss_on_normalized(preds_n_val, targets_n_val) if 'mae' in loss_type else pixel_mse_loss(preds_n_val, targets_n_val)
                total_loss_v = main_loss_v + (lambda_gdl * sobel_gradient_loss(targets_n_val, preds_n_val, device) if 'gdl' in loss_type else 0)
                preds_d_val = denormalize_target_z_score(preds_n_val, target_original_mean, target_original_std)
                targets_d_val = denormalize_target_z_score(targets_n_val, target_original_mean, target_original_std)
                epoch_val_loss_n.append(total_loss_v.item())
                epoch_val_mae_d.append(torch.mean(torch.abs(preds_d_val - targets_d_val)).item())

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
        
        scheduler.step()     

    print(f"Training of {run_id} done. Best Val MAE(D): {best_val_mae_denorm:.4f} @ {weights_path}")
    
    # Testovanie (na statických dátach)
    if os.path.exists(weights_path) and test_loader is not None: # Pridaná kontrola pre test_loader
        print(f"\nTesting with best weights for {run_id}...")
        net.load_state_dict(torch.load(weights_path, map_location=device)) # Pridané map_location
        net.eval()
        test_mae_d_list = []
        with torch.no_grad():
            for i, batch_data_test in enumerate(test_loader):
                if batch_data_test[0] is None or batch_data_test[1] is None: continue
                inputs_n_test, targets_n_test = batch_data_test[0].to(device), batch_data_test[1].to(device)
                preds_n_test = net(inputs_n_test)
                preds_d_test = denormalize_target_z_score(preds_n_test, target_original_mean, target_original_std)
                targets_d_test = denormalize_target_z_score(targets_n_test, target_original_mean, target_original_std)
                mae_per_image_in_batch = torch.mean(torch.abs(preds_d_test - targets_d_test), dim=(1,2,3))
                test_mae_d_list.extend(mae_per_image_in_batch.cpu().numpy())
                
                if i == 0 and len(inputs_n_test)>0: 
                    j=0
                    pred_img_d=preds_d_test[j].cpu().numpy().squeeze()
                    lbl_img_d=targets_d_test[j].cpu().numpy().squeeze()
                    
                    # Príprava vstupného obrázka na zobrazenie
                    input_to_show_np = inputs_n_test[j].cpu().numpy().squeeze() # (C, H, W) or (H,W)
                    
                    title_in_slovak = ""
                    vmin_in_show, vmax_in_show = None, None # Pre automatické škálovanie, ak nie je direct_minmax

                    if input_processing_type=='sincos':
                        # Zobrazíme len prvý kanál (napr. sin(phi)) pre jednoduchosť
                        input_to_show_final_np = input_to_show_np[0,:,:] if input_to_show_np.ndim == 3 else input_to_show_np
                        title_in_slovak = "Vstup (sin(φ) kanál, norm.)"
                        vmin_in_show,vmax_in_show = (-1,1)
                    else: # direct_minmax
                        # Denormalizácia vstupu
                        if input_original_min_max:
                            # Konverzia na torch tensor pre denormalizačnú funkciu, ak ešte nie je
                            input_to_show_tensor = inputs_n_test[j].cpu() # Už by mal byť tensor
                            denorm_input_tensor = denormalize_input_minmax_from_minus_one_one(
                                input_to_show_tensor, 
                                input_original_min_max[0], 
                                input_original_min_max[1]
                            )
                            input_to_show_final_np = denorm_input_tensor.numpy().squeeze()
                            title_in_slovak = "Vstup (Wrapped, denorm.)"
                            # vmin/vmax pre denormalizovaný obraz sa nastavia automaticky imshow
                        else: # Fallback, ak by chýbali globálne štatistiky pre vstup
                            input_to_show_final_np = input_to_show_np.squeeze() # Zobrazí normalizovaný
                            title_in_slovak = "Vstup (Wrapped, norm. [-1,1])"
                            vmin_in_show,vmax_in_show = (-1,1)


                    fig_test_vis, axes_test_vis = plt.subplots(1,3,figsize=(18,6)); 
                    fig_test_vis.suptitle(f"Testovacia vizualizácia",fontsize=16)
                    
                    im0 = axes_test_vis[0].imshow(input_to_show_final_np,cmap='gray',vmin=vmin_in_show,vmax=vmax_in_show); 
                    axes_test_vis[0].set_title(title_in_slovak, fontsize=14); 
                    fig_test_vis.colorbar(im0, ax=axes_test_vis[0])
                    
                    im1 = axes_test_vis[1].imshow(lbl_img_d,cmap='gray'); 
                    axes_test_vis[1].set_title("Referenčný obraz (denorm.)", fontsize=14); 
                    fig_test_vis.colorbar(im1, ax=axes_test_vis[1])
                    
                    im2 = axes_test_vis[2].imshow(pred_img_d,cmap='gray'); 
                    axes_test_vis[2].set_title("Predikcia (denorm.)", fontsize=14); 
                    fig_test_vis.colorbar(im2, ax=axes_test_vis[2])
                    
                    plt.tight_layout(rect=[0,0,1,0.95]); 
                    plt.savefig(f'vis_{run_id}.png'); 
                    plt.close(fig_test_vis)
                    try:
                        # Ukladanie denormalizovaného vstupu ako TIFF
                        tiff.imwrite(f'input_denorm_{run_id}.tiff', input_to_show_final_np.astype(np.float32))
                        tiff.imwrite(f'pred_{run_id}.tiff', pred_img_d.astype(np.float32))
                        tiff.imwrite(f'gt_{run_id}.tiff', lbl_img_d.astype(np.float32))
                    except Exception as e_tiff:
                        print(f"Chyba pri ukladaní testovacích TIFF: {e_tiff}")


        avg_test_mae_d = np.mean(test_mae_d_list) if test_mae_d_list else float('nan')
        print(f"  Average Test MAE (Denorm): {avg_test_mae_d:.6f}")
        with open(f"metrics_{run_id}.txt", "w") as f:
            f.write(f"Run ID: {run_id}\nBest Val MAE (Denorm): {best_val_mae_denorm:.6f}\nTest MAE (Denorm): {avg_test_mae_d:.6f}\n")
    elif not os.path.exists(weights_path): 
        print(f"No weights found for {run_id} at {weights_path} to test.")
    elif test_loader is None:
        print(f"Test loader not provided for {run_id}, skipping testing.")


    # Kreslenie grafov straty a MAE
    fig_curves, (ax_loss, ax_mae) = plt.subplots(1,2,figsize=(12,5))
    
    ax_loss.plot(train_loss_hist,label='Tréningová strata (N)'); # ZMENENÝ LABEL
    ax_loss.plot(val_loss_hist,label='Validačná strata (N)')  # ZMENENÝ LABEL
    ax_loss.set_title('Priebeh normalizovanej straty', fontsize=16); 
    ax_loss.set_xlabel('Epocha', fontsize=12); 
    ax_loss.set_ylabel('Strata (N)', fontsize=12); 
    ax_loss.legend(); ax_loss.grid(True)
    
    ax_mae.plot(train_mae_denorm_hist,label='Tréningová MAE (D)'); 
    ax_mae.plot(val_mae_denorm_hist,label='Validačná MAE (D)')
    ax_mae.set_title('Priebeh denormalizovanej MAE', fontsize=16); 
    ax_mae.set_xlabel('Epocha', fontsize=12); 
    ax_mae.set_ylabel('MAE (D)', fontsize=12); 
    ax_mae.legend(); ax_mae.grid(True)
    
    plt.tight_layout(); 
    plt.savefig(f'curves_{run_id}.png'); 
    plt.close(fig_curves)
    
    return {"best_val_mae_denorm": best_val_mae_denorm, 
            "test_mae_denorm": avg_test_mae_d if os.path.exists(weights_path) and test_loader is not None else float('nan')}


# --- Hlavná funkcia pre spustenie experimentov ---
def spusti_experimenty_s_dynamickym_treningom():
    # --- CESTY K DATASETOM (musia zodpovedať výstupu Skriptu 1) ---
    output_base_dir_from_script1 = "split_dataset_tiff_for_dynamic_v_stratified_final" # MUSÍ SA ZHODOVAŤ!

    path_static_ref_train = os.path.join(output_base_dir_from_script1, "static_ref_train_dataset")
    path_static_valid = os.path.join(output_base_dir_from_script1, "static_valid_dataset")
    path_static_test = os.path.join(output_base_dir_from_script1, "static_test_dataset")
    path_dynamic_train_source = os.path.join(output_base_dir_from_script1, "train_dataset_source_for_dynamic_generation", "images")

    # 1. Výpočet globálnych normalizačných štatistík zo statického referenčného tréningového setu
    print("--- VÝPOČET GLOBÁLNYCH NORMALIZAČNÝCH ŠTATISTÍK ---")
    if not os.path.exists(path_static_ref_train):
        raise FileNotFoundError(f"Adresár pre statický referenčný tréningový set nebol nájdený: {path_static_ref_train}. Spustite najprv skript na prípravu datasetov.")
        
    GLOBAL_UNWRAPPED_MEAN, GLOBAL_UNWRAPPED_STD = get_global_target_mean_std_stats(path_static_ref_train, "Global Unwrapped Target (from Ref Train)")
    GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX = get_global_input_min_max_stats(path_static_ref_train, "Global Wrapped Input (from Ref Train)")
    
    print(f"\nGlobálne normalizačné štatistiky (z referenčného tréningového setu):")
    print(f"  Wrapped Input: Min={GLOBAL_WRAPPED_MIN:.4f}, Max={GLOBAL_WRAPPED_MAX:.4f}")
    print(f"  Unwrapped Target: Mean={GLOBAL_UNWRAPPED_MEAN:.4f}, Std={GLOBAL_UNWRAPPED_STD:.4f}")
    if GLOBAL_UNWRAPPED_STD < 1e-6: raise ValueError("Globálna Std pre unwrapped dáta je nula!")
    if GLOBAL_WRAPPED_MIN == GLOBAL_WRAPPED_MAX: raise ValueError("Globálne Min a Max pre wrapped dáta sú rovnaké!")

    # Kontrola integrity pre statické sady (ak chcete)
    # for ds_path_check in [path_static_valid, path_static_test, path_static_ref_train]:
    #     check_dataset_integrity(ds_path_check) # Táto funkcia by potrebovala úpravu pre nové názvy súborov
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tento parameter môžete meniť pre experimenty
    # Ak `0` bolo najrýchlejšie, skúste teraz s dynamickým generovaním, či `>0` pomôže
    # Odporúčané hodnoty: 0, 2, 4, alebo os.cpu_count() // 2
    num_train_data_workers = 1 # !!! EXPERIMENTUJTE S TÝMTO !!!
    num_eval_data_workers = 1  # Pre valid/test zvyčajne stačí 0 alebo malá hodnota

    # --- Parametre simulácie (rovnaké ako v Skripte 1) ---
    simulation_param_ranges_config = {
        "n_strips_param": (7, 8), 
        "original_image_influence_param": (0.3, 0.5),
        "phase_noise_std_param": (0.024, 0.039),
        "smooth_original_image_sigma_param": (0.2, 0.5),
        "poly_scale_param": (0.02, 0.1), 
        "CURVATURE_AMPLITUDE_param": (1.4, 2.0), 
        "background_offset_d_param": (-24.8, -6.8), 
        "tilt_angle_deg_param": (-5.0, 17.0) 
    }
    amplify_ab_fixed_config_val = 1.0 
    # ----------------------------------------------------

    # --- DEFINÍCIA EXPERIMENTOV (zostáva rovnaká ako vo vašom tréningovom skripte) ---
    experiments = [
        # {
        #     "run_id_suffix": "AUTO_GENERATED", 
        #     "encoder_name": "resnet34", "encoder_weights": "imagenet",  
        #     "input_processing": "direct_minmax", "augmentation_strength": "light", 
        #     "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
        #     "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30, 
        #     "cosine_T_max": 120, "cosine_eta_min": 1e-7,
        #     "wd": 1e-4
        # },
        # {
        #     "run_id_suffix": "AUTO_GENERATED",
        #     "encoder_name": "resnet34", "encoder_weights": "imagenet",
        #     "input_processing": "direct_minmax", "augmentation_strength": "strong", 
        #     "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
        #     "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30,
        #     "cosine_T_max": 120, "cosine_eta_min": 1e-7,
        #     "wd": 1e-4
        # },
        # {
        #     "run_id_suffix": "AUTO_GENERATED",
        #     "encoder_name": "resnet34", "encoder_weights": "imagenet",
        #     "input_processing": "direct_minmax", "augmentation_strength": "medium", 
        #     "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
        #     "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30,
        #     "cosine_T_max": 120, "cosine_eta_min": 1e-7,
        #     "wd": 1e-4
        # },
        # {
        #     "run_id_suffix": "AUTO_GENERATED",
        #     "encoder_name": "resnet34", "encoder_weights": "imagenet",
        #     "input_processing": "direct_minmax", "augmentation_strength": "medium", 
        #     "loss_type": "mae_gdl", "lambda_gdl": 0.3, 
        #     "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30,
        #     "cosine_T_max": 120, "cosine_eta_min": 1e-7,        
        #     "wd": 1e-4
        # },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong", 
            "loss_type": "mae_gdl", "lambda_gdl": 0.3, 
            "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30,
            "cosine_T_max": 120, "cosine_eta_min": 1e-7,
            "wd": 1e-4
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "medium", 
            "loss_type": "mae_gdl", "lambda_gdl": 0.3, 
            "lr": 5e-4, "bs": 8, "epochs": 120, "es_pat": 30,
            "cosine_T_max": 120, "cosine_eta_min": 1e-7,
            "wd": 1e-4
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong", 
            "loss_type": "mae_gdl", "lambda_gdl": 0.3, 
            "lr": 5e-4, "bs": 8, "epochs": 120, "es_pat": 30,
            "cosine_T_max": 120, "cosine_eta_min": 1e-7,
            "wd": 1e-4
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "strong", 
            "loss_type": "mae_gdl", "lambda_gdl": 0.1, 
            "lr": 5e-4, "bs": 8, "epochs": 120, "es_pat": 30,
            "cosine_T_max": 120, "cosine_eta_min": 1e-7,
            "wd": 1e-4
        },
        {
            "run_id_suffix": "AUTO_GENERATED",
            "encoder_name": "resnet34", "encoder_weights": "imagenet",
            "input_processing": "direct_minmax", "augmentation_strength": "medium", 
            "loss_type": "mae_gdl", "lambda_gdl": 0.5, 
            "lr": 1e-3, "bs": 8, "epochs": 120, "es_pat": 30,
            "cosine_T_max": 120, "cosine_eta_min": 1e-7,
            "wd": 1e-4
        },
        # ... (pridajte ďalšie konfigurácie experimentov podľa potreby) ...
    ]

    all_run_results = []
    for cfg_original in experiments:
        cfg = cfg_original.copy()

        # --- Dynamické generovanie run_id_suffix (zostáva rovnaké) ---
        encoder_short_map = {"resnet34": "R34", "resnet18": "R18"} 
        enc_name_part = encoder_short_map.get(cfg["encoder_name"], cfg["encoder_name"])
        enc_weights_part = ""
        if cfg["encoder_weights"] is None or (isinstance(cfg["encoder_weights"], str) and cfg["encoder_weights"].lower() == "none"):
            enc_weights_part = "scratch"
        elif isinstance(cfg["encoder_weights"], str) and cfg["encoder_weights"].lower() == "imagenet":
            enc_weights_part = "imgnet"
        enc_part = f"{enc_name_part}{enc_weights_part}"
        inp_proc_part = cfg["input_processing"] if cfg["input_processing"] != "direct_minmax" else "direct"
        loss_part = "MAE" if "mae" in cfg["loss_type"] else ("MSE" if "mse" in cfg["loss_type"] else cfg["loss_type"])
        gdl_val = cfg.get("lambda_gdl", 0.0)
        gdl_part = f"GDL{str(gdl_val).replace('.', 'p')}" if gdl_val > 0 and ('gdl' in cfg["loss_type"] or loss_part) else ""
        aug_strength = cfg["augmentation_strength"]
        aug_part = f"Aug{aug_strength.capitalize()}" if aug_strength and aug_strength.lower() != 'none' else "AugNone"
        lr_val_str = f"{cfg['lr']:.0e}"; lr_part = f"LR{lr_val_str.replace('-', 'm')}"
        epochs_part = f"Ep{cfg['epochs']}"; es_pat_part = f"ESp{cfg['es_pat']}"
        cosine_T_max_part = f"Tmax{cfg['cosine_T_max']}"
        eta_min_val = cfg.get('cosine_eta_min')
        if eta_min_val is None: cosine_eta_min_part = "EtaMinDef"
        elif eta_min_val == 0: cosine_eta_min_part = "EtaMin0"
        else: cosine_eta_min_part = f"EtaMin{eta_min_val:.0e}".replace('-', 'm')
        
        wd_val = cfg.get("wd", 0.0) # Získanie weight decay, predvolená hodnota 0.0 ak nie je definovaná
        wd_part = f"WD{wd_val:.0e}".replace('-', 'm') if wd_val > 0 else "WD0"

        bs_part = f"bs{cfg['bs']}"
        parts = [part for part in [enc_part, inp_proc_part, loss_part, gdl_part, aug_part, lr_part, wd_part, epochs_part, es_pat_part, cosine_T_max_part, cosine_eta_min_part, bs_part] if part] 
        
        generated_suffix = "_".join(parts)
        # Použije sa cfg_original["run_id_suffix"] ak je definovaný a nie je "AUTO_GENERATED", inak sa použije generovaný
        if "run_id_suffix" in cfg_original and cfg_original["run_id_suffix"] != "AUTO_GENERATED":
            run_id_final = cfg_original["run_id_suffix"]
        else:
            run_id_final = generated_suffix
        cfg['run_id_suffix'] = run_id_final # Uložíme finálny run_id do cfg pre konzistentnosť

        print(f"\n\n{'='*25} EXPERIMENT: {cfg['run_id_suffix']} {'='*25}")
        
        # --- Vytvorenie DataLoaders ---
        # Dynamický tréningový DataLoader
        train_source_filepaths_for_loader = glob.glob(os.path.join(path_dynamic_train_source, "*.tif*"))
        if not train_source_filepaths_for_loader:
            raise FileNotFoundError(f"Nenašli sa žiadne zdrojové obrázky pre dynamický tréning v: {path_dynamic_train_source}")
            
        train_dynamic_ds = DynamicTrainPhaseDataset(
            source_image_filepaths=train_source_filepaths_for_loader,
            simulation_param_ranges=simulation_param_ranges_config, # Globálne pre simuláciu
            amplify_ab_fixed_value=amplify_ab_fixed_config_val,    # Globálne pre simuláciu
            input_processing_type=cfg["input_processing"],
            norm_stats_input_global=(GLOBAL_WRAPPED_MIN, GLOBAL_WRAPPED_MAX),
            norm_stats_target_global=(GLOBAL_UNWRAPPED_MEAN, GLOBAL_UNWRAPPED_STD),
            augmentation_strength=cfg["augmentation_strength"], # Augmentácia pre tréning
            target_img_size=(512,512) # Predpokladáme, že zdrojové sú už tejto veľkosti
        )
        train_loader = DataLoader(train_dynamic_ds, batch_size=cfg["bs"], shuffle=True, 
                                  num_workers=num_train_data_workers, pin_memory=True, 
                                  persistent_workers=True if num_train_data_workers > 0 else False,
                                  collate_fn=collate_fn_skip_none) # Pridaný collate_fn

        # Statický validačný DataLoader
        val_ds = StaticPhaseDataset(path_static_valid,
                                    cfg["input_processing"],
                                    (GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX),
                                    (GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD))
        val_loader = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False, 
                                num_workers=num_eval_data_workers, pin_memory=True,
                                collate_fn=collate_fn_skip_none)

        # Statický testovací DataLoader
        test_ds = StaticPhaseDataset(path_static_test,
                                     cfg["input_processing"],
                                     (GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX),
                                     (GLOBAL_UNWRAPPED_MEAN,GLOBAL_UNWRAPPED_STD))
        test_loader = DataLoader(test_ds, batch_size=cfg["bs"], shuffle=False, 
                                 num_workers=num_eval_data_workers, pin_memory=True,
                                 collate_fn=collate_fn_skip_none)
        
        run_id_final = f"{cfg['run_id_suffix']}_bs{cfg['bs']}"
        
        exp_results = run_training_session(
            run_id=run_id_final, device=device, num_epochs=cfg["epochs"],
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            target_original_mean=GLOBAL_UNWRAPPED_MEAN, target_original_std=GLOBAL_UNWRAPPED_STD,
            input_original_min_max=(GLOBAL_WRAPPED_MIN,GLOBAL_WRAPPED_MAX) if cfg["input_processing"]=='direct_minmax' else None,
            encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
            input_processing_type=cfg["input_processing"], loss_type=cfg["loss_type"],
            lambda_gdl=cfg.get("lambda_gdl",0.0), learning_rate=cfg["lr"], 
            weight_decay=cfg.get("wd", 1e-4), # Použitie weight_decay z cfg
            cosine_T_max=cfg["cosine_T_max"], 
            cosine_eta_min=cfg.get("cosine_eta_min"),
            min_lr=1e-7, 
            early_stopping_patience=cfg["es_pat"], 
            augmentation_strength_train=cfg["augmentation_strength"] 
        )
        all_run_results.append({"run_id": run_id_final, "config": cfg, "metrics": exp_results})

    print("\n\n" + "="*30 + " SÚHRN VÝSLEDKOV " + "="*30)
    # ... (zvyšok výpisu výsledkov zostáva rovnaký) ...
    for summary in all_run_results:
        print(f"Run: {summary['run_id']}")
        print(f"  Best Val MAE (D): {summary['metrics'].get('best_val_mae_denorm', 'N/A'):.4f}")
        print(f"  Test MAE (D):     {summary['metrics'].get('test_mae_denorm', 'N/A'):.4f}")
        print("-" * 70)
    print(f"--- VŠETKY EXPERIMENTY DOKONČENÉ ---")


if __name__ == '__main__':
    # Predpokladáme, že Skript 1 (na prípravu datasetov) už zbehol
    # a vytvoril adresárovú štruktúru v 'split_dataset_tiff_for_dynamic_v_stratified_final'
    
    # Dôležité: Načítanie torch až tu, aby np.random.seed() v Skripte 1 neovplyvnil PyTorch seed, ak by ste ich spájali.
    # Ale keďže sú to separátne skripty, je to menej kritické.
    import torch 

    # Nastavenie náhodného semena pre PyTorch pre reprodukovateľnosť tréningu
    # Toto by malo byť nastavené pred vytváraním modelu a optimalizátora, ak chcete konzistentné váhy na štarte
    torch_seed = 42
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        # Nasledujúce dva riadky môžu spomaliť tréning, ak ich potrebujete pre presnú reprodukovateľnosť
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    spusti_experimenty_s_dynamickym_treningom()