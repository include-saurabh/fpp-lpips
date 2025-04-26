import os
import numpy as np
from PIL import Image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress specific warnings if needed (e.g., from skimage)
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')
warnings.filterwarnings("ignore", category=FutureWarning) # Often from libraries updating


# --- Configuration ---
# <<< KEEP YOUR ORIGINAL FOLDER PATHS HERE >>>
ORIGINAL_FOLDER = r'C:\Users\CSE IIT BHILAI\Facial-Privacy-Protection\results\img_quality_check final\celeborig'
APPROACH2_FOLDER = r'C:\Users\CSE IIT BHILAI\Facial-Privacy-Protection\results\img_quality_check final\lpips'
APPROACH1_FOLDER = r'C:\Users\CSE IIT BHILAI\Facial-Privacy-Protection\results\img_quality_check final\origrp'
OUTPUT_CSV = 'image_metrics_comparison_normalized.csv' # Updated CSV name

# Check if GPU is available for LPIPS, otherwise use CPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"Using device: {device}")

# Load LPIPS model (choose 'alex' or 'vgg', 'alex' is faster)
# Loads the model onto the chosen device
try:
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    print("LPIPS model loaded successfully.")
except Exception as e:
    print(f"Error loading LPIPS model: {e}")
    print("LPIPS calculations will fail. Ensure the 'lpips' package is installed correctly.")
    loss_fn_alex = None # Set to None to handle errors later

# --- Helper Functions ---

def preprocess_image_for_lpips(img_path):
    """Loads an image, converts to tensor, normalizes for LPIPS."""
    img = lpips.load_image(img_path) # Loads image as RGB float32 on [0,1]
    img_tensor = lpips.im2tensor(img).to(device) # Converts to CxHxW tensor, scales to [-1,+1]
    return img_tensor

def calculate_metrics(img_orig_path, img_pert_path):
    """Calculates PSNR, SSIM, and LPIPS between two images."""
    results = {'psnr': np.nan, 'ssim': np.nan, 'lpips': np.nan} # Default to NaN
    try:
        # Load images for PSNR/SSIM (using PIL/Numpy for consistency)
        img_orig_pil = Image.open(img_orig_path).convert('RGB')
        img_pert_pil = Image.open(img_pert_path).convert('RGB')

        # Ensure images have the same size (resize perturbed if needed, using original's size)
        if img_orig_pil.size != img_pert_pil.size:
            print(f"Warning: Resizing perturbed image {os.path.basename(img_pert_path)} to match original {os.path.basename(img_orig_path)} size {img_orig_pil.size}")
            img_pert_pil = img_pert_pil.resize(img_orig_pil.size, Image.Resampling.LANCZOS) # Or BILINEAR

        img_orig_np = np.array(img_orig_pil).astype(np.float32)
        img_pert_np = np.array(img_pert_pil).astype(np.float32)

        # --- PSNR ---
        # Check for identical images first to avoid potential log(0) issues
        if np.array_equal(img_orig_np, img_pert_np):
             results['psnr'] = float('inf') # Assign infinity for identical images
        else:
             # Calculate MSE, ensuring it's not zero before calculating PSNR
             mse = np.mean((img_orig_np - img_pert_np) ** 2)
             if mse == 0:
                 results['psnr'] = float('inf') # Should be caught by array_equal, but double-check
             else:
                 # data_range is max possible pixel value (255 for uint8 equivalent)
                 # Using skimage's psnr function handles the calculation robustly
                 try:
                     # skimage expects uint8 or float in [0,1] range for default data_range=1
                     # Since we have float32 [0, 255], specify data_range=255
                      results['psnr'] = psnr(img_orig_np, img_pert_np, data_range=255)
                 except ValueError as psnr_err:
                     print(f"PSNR calculation error for {os.path.basename(img_pert_path)}: {psnr_err}. MSE was {mse}. Setting PSNR to NaN.")
                     results['psnr'] = np.nan


        # --- SSIM ---
        # For multichannel (color) images, set channel_axis appropriately
        # Ensure win_size is odd and less than or equal to image dimensions
        min_dim = min(img_orig_np.shape[0], img_orig_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1) # Ensure win_size is odd and <= min dim
        if win_size < 3: # win_size must be >= 3
             print(f"Warning: Image dimension too small for default SSIM win_size ({min_dim}). Skipping SSIM for {os.path.basename(img_pert_path)}.")
             results['ssim'] = np.nan
        else:
            try:
                 results['ssim'] = ssim(img_orig_np, img_pert_np, data_range=255, channel_axis=-1, win_size=win_size, K1=0.01, K2=0.03) # Explicit K values often used
            except ValueError as ssim_err:
                 print(f"SSIM calculation error for {os.path.basename(img_pert_path)}: {ssim_err}. Setting SSIM to NaN.")
                 results['ssim'] = np.nan

        # --- LPIPS ---
        if loss_fn_alex is not None: # Check if LPIPS model loaded
            # Preprocess images specifically for LPIPS
            img_orig_tensor = preprocess_image_for_lpips(img_orig_path)
            img_pert_tensor = preprocess_image_for_lpips(img_pert_path)

            # Basic check, LPIPS might handle small diffs depending on network
            if img_orig_tensor.shape != img_pert_tensor.shape:
                # This shouldn't happen if PIL resize worked, but good to check
                print(f"Tensor shape mismatch for LPIPS: {img_orig_tensor.shape} vs {img_pert_tensor.shape}. Skipping LPIPS for {os.path.basename(img_orig_path)}")
                results['lpips'] = np.nan # Or None
            else:
                with torch.no_grad(): # Important: disable gradient calculation
                    dist = loss_fn_alex(img_orig_tensor, img_pert_tensor).item()
                results['lpips'] = dist
        else:
            results['lpips'] = np.nan # LPIPS model failed to load

    except FileNotFoundError:
        print(f"Error: File not found. Original: {img_orig_path}, Perturbed: {img_pert_path}")
        # results remain NaN as initialized
    except Exception as e:
        print(f"Error processing {os.path.basename(img_orig_path)} vs {os.path.basename(img_pert_path)}: {e}")
        # Set metrics to NaN upon error to allow partial results
        results = {'psnr': np.nan, 'ssim': np.nan, 'lpips': np.nan}

    return results


# --- Main Processing Loop ---
results_list = []
original_files = []
if os.path.isdir(ORIGINAL_FOLDER):
    original_files = sorted([f for f in os.listdir(ORIGINAL_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
else:
    print(f"Error: Original folder not found or is not a directory: {ORIGINAL_FOLDER}")
    exit()


print(f"Found {len(original_files)} images in {ORIGINAL_FOLDER}.")

# Check other folders exist
if not os.path.isdir(APPROACH1_FOLDER):
    print(f"Error: Approach 1 folder not found or is not a directory: {APPROACH1_FOLDER}")
    exit()
if not os.path.isdir(APPROACH2_FOLDER):
    print(f"Error: Approach 2 folder not found or is not a directory: {APPROACH2_FOLDER}")
    exit()


for filename in tqdm(original_files, desc="Processing Images"):
    original_path = os.path.join(ORIGINAL_FOLDER, filename)
    approach1_path = os.path.join(APPROACH1_FOLDER, filename)
    approach2_path = os.path.join(APPROACH2_FOLDER, filename)

    # Check if corresponding files exist before calling calculate_metrics
    if not os.path.exists(approach1_path):
        print(f"Warning: Corresponding file missing in {APPROACH1_FOLDER} for {filename}. Skipping.")
        continue
    if not os.path.exists(approach2_path):
        print(f"Warning: Corresponding file missing in {APPROACH2_FOLDER} for {filename}. Skipping.")
        continue

    # Calculate metrics for Approach 1
    metrics_a1 = calculate_metrics(original_path, approach1_path)

    # Calculate metrics for Approach 2
    metrics_a2 = calculate_metrics(original_path, approach2_path)

    # Only append if *all* metrics were calculated successfully for *both* approaches
    # (We handle NaNs later, but let's store the raw results first)
    results_list.append({
        'filename': filename,
        'psnr_a1': metrics_a1['psnr'],
        'ssim_a1': metrics_a1['ssim'],
        'lpips_a1': metrics_a1['lpips'],
        'psnr_a2': metrics_a2['psnr'],
        'ssim_a2': metrics_a2['ssim'],
        'lpips_a2': metrics_a2['lpips'],
    })

# --- Analysis ---
if not results_list:
    print("No results were generated. Check folder paths and file existence.")
    exit()

df = pd.DataFrame(results_list)

# --- !! IMPORTANT: Handle potential Inf values in PSNR !! ---
# Replace Inf with a very large number or NaN before normalization/analysis
# Using NaN is often safer as it gets excluded from averages correctly.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle potential NaN values if errors occurred OR Inf was replaced
rows_before_drop = len(df)
df.dropna(inplace=True) # Drop rows where any metric calculation failed or resulted in NaN/Inf
rows_after_drop = len(df)
print(f"\nDropped {rows_before_drop - rows_after_drop} rows due to missing/invalid metric values (NaN/Inf).")


if df.empty:
    print("DataFrame is empty after dropping NaN/Inf values. Cannot proceed with analysis.")
    exit()


# --- Normalization (Min-Max Scaling to [0, 1]) ---
print("\nNormalizing metrics...")
metrics_to_normalize = ['psnr', 'ssim', 'lpips']
epsilon = 1e-9 # Small value to prevent division by zero if min == max

for metric in metrics_to_normalize:
    col_a1 = f'{metric}_a1'
    col_a2 = f'{metric}_a2'
    norm_col_a1 = f'{metric}_a1_norm'
    norm_col_a2 = f'{metric}_a2_norm'

    # Find min and max across both approaches for the current metric
    min_val = df[[col_a1, col_a2]].min().min()
    max_val = df[[col_a1, col_a2]].max().max()
    value_range = max_val - min_val

    # Normalize Approach 1
    if abs(value_range) < epsilon: # Handle case where all values are the same
         df[norm_col_a1] = 0.5 # Or 0 or 1, depending on desired behavior
    else:
         df[norm_col_a1] = (df[col_a1] - min_val) / value_range

    # Normalize Approach 2
    if abs(value_range) < epsilon:
         df[norm_col_a2] = 0.5
    else:
         df[norm_col_a2] = (df[col_a2] - min_val) / value_range

    # --- Special Handling for LPIPS ---
    # Lower LPIPS is better. To make the normalized score consistent
    # where higher is better (like PSNR/SSIM), invert the normalized LPIPS.
    # So, 1.0 becomes the best possible score (lowest LPIPS), 0.0 the worst.
    if metric == 'lpips':
        df[norm_col_a1] = 1.0 - df[norm_col_a1]
        df[norm_col_a2] = 1.0 - df[norm_col_a2]
        print(f"  - Normalized {metric} (inverted: higher is better)")
    else:
        print(f"  - Normalized {metric} (higher is better)")


# --- Save results to CSV (including normalized values) ---
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f') # Save with more precision
print(f"\nMetrics (including normalized) saved to {OUTPUT_CSV}")

# --- Calculate and print average metrics (Original and Normalized) ---
avg_metrics = df.mean(numeric_only=True)

print("\n--- Average Metrics (Original Scale) ---")
print(f"Approach 1 - PSNR Avg:   {avg_metrics.get('psnr_a1', 'N/A'):.4f} dB")
print(f"Approach 2 - PSNR Avg:   {avg_metrics.get('psnr_a2', 'N/A'):.4f} dB")
print("-" * 20)
print(f"Approach 1 - SSIM Avg:   {avg_metrics.get('ssim_a1', 'N/A'):.4f}")
print(f"Approach 2 - SSIM Avg:   {avg_metrics.get('ssim_a2', 'N/A'):.4f}")
print("-" * 20)
print(f"Approach 1 - LPIPS Avg:  {avg_metrics.get('lpips_a1', 'N/A'):.4f} (Lower is better)")
print(f"Approach 2 - LPIPS Avg:  {avg_metrics.get('lpips_a2', 'N/A'):.4f} (Lower is better)")
print("-" * 20)

print("\n--- Average Metrics (Normalized Scale [0, 1]) ---")
print(f"Approach 1 - PSNR Norm Avg:  {avg_metrics.get('psnr_a1_norm', 'N/A'):.4f}")
print(f"Approach 2 - PSNR Norm Avg:  {avg_metrics.get('psnr_a2_norm', 'N/A'):.4f}")
print("-" * 20)
print(f"Approach 1 - SSIM Norm Avg:  {avg_metrics.get('ssim_a1_norm', 'N/A'):.4f}")
print(f"Approach 2 - SSIM Norm Avg:  {avg_metrics.get('ssim_a2_norm', 'N/A'):.4f}")
print("-" * 20)
# Note: For normalized LPIPS, higher is now better due to inversion
print(f"Approach 1 - LPIPS Norm Avg: {avg_metrics.get('lpips_a1_norm', 'N/A'):.4f} (Higher is better)")
print(f"Approach 2 - LPIPS Norm Avg: {avg_metrics.get('lpips_a2_norm', 'N/A'):.4f} (Higher is better)")
print("-" * 20)


# --- Interpretation based on original averages ---
# (Interpretation section remains unchanged, based on original scales)
print("\n--- Interpretation (Based on Original Averages) ---")
# Check if keys exist before accessing, in case calculation failed entirely for a metric
if 'psnr_a1' in avg_metrics and 'psnr_a2' in avg_metrics:
    if avg_metrics['psnr_a2'] > avg_metrics['psnr_a1']:
        print("PSNR: Approach 2 has higher average PSNR (better fidelity).")
    elif avg_metrics['psnr_a1'] > avg_metrics['psnr_a2']:
        print("PSNR: Approach 1 has higher average PSNR (better fidelity).")
    else:
        print("PSNR: Approaches have equal average PSNR.")
else:
    print("PSNR: Could not compare averages (missing data).")

if 'ssim_a1' in avg_metrics and 'ssim_a2' in avg_metrics:
    if avg_metrics['ssim_a2'] > avg_metrics['ssim_a1']:
        print("SSIM: Approach 2 has higher average SSIM (better structural similarity).")
    elif avg_metrics['ssim_a1'] > avg_metrics['ssim_a2']:
        print("SSIM: Approach 1 has higher average SSIM (better structural similarity).")
    else:
        print("SSIM: Approaches have equal average SSIM.")
else:
    print("SSIM: Could not compare averages (missing data).")

if 'lpips_a1' in avg_metrics and 'lpips_a2' in avg_metrics:
    if avg_metrics['lpips_a2'] < avg_metrics['lpips_a1']:
        print("LPIPS: Approach 2 has lower average LPIPS (better perceptual similarity).")
    elif avg_metrics['lpips_a1'] < avg_metrics['lpips_a2']:
         print("LPIPS: Approach 1 has lower average LPIPS (better perceptual similarity).")
    else:
        print("LPIPS: Approaches have equal average LPIPS.")
else:
    print("LPIPS: Could not compare averages (missing data).")


# --- Visualization (Using NORMALIZED Values) --- # <<< MODIFIED SECTION START >>>

print("\nGenerating visualizations using NORMALIZED metric scales...")
sns.set_theme(style="whitegrid")

# --- Create a melted dataframe for NORMALIZED values ---
df_melt_norm = pd.melt(df, id_vars=['filename'],
                       value_vars=[f'{m}_{a}_norm' for m in metrics_to_normalize for a in ['a1', 'a2']],
                       var_name='metric_approach_norm', value_name='value_norm')
# Split metric_approach_norm into metric and approach columns
# Expecting format like 'psnr_a1_norm'
split_cols = df_melt_norm['metric_approach_norm'].str.split('_', expand=True)
df_melt_norm['metric'] = split_cols[0]
df_melt_norm['approach'] = split_cols[1].str.upper().replace({'A1': 'Approach 1', 'A2': 'Approach 2'})
# df_melt_norm['norm_suffix'] = split_cols[2] # We don't really need this column


# 1. Box Plots / Violin Plots for Distribution Comparison (NORMALIZED)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Metric Distributions (Normalized Scale [0,1]): Approach 1 vs Approach 2', fontsize=16) # <-- Title updated

# PSNR Plot (Normalized)
sns.boxplot(ax=axes[0], data=df_melt_norm[df_melt_norm['metric'] == 'psnr'], x='approach', y='value_norm', palette="Blues") # <-- Use df_melt_norm, y='value_norm'
axes[0].set_title('Normalized PSNR Distribution') # <-- Title updated
axes[0].set_xlabel('Approach')
axes[0].set_ylabel('Normalized PSNR (Higher is Better)') # <-- Label updated
axes[0].set_ylim(-0.05, 1.05) # Set Y axis limits for [0,1] scale

# SSIM Plot (Normalized)
sns.boxplot(ax=axes[1], data=df_melt_norm[df_melt_norm['metric'] == 'ssim'], x='approach', y='value_norm', palette="Greens") # <-- Use df_melt_norm, y='value_norm'
axes[1].set_title('Normalized SSIM Distribution') # <-- Title updated
axes[1].set_xlabel('Approach')
axes[1].set_ylabel('Normalized SSIM (Higher is Better)') # <-- Label updated
axes[1].set_ylim(-0.05, 1.05) # Set Y axis limits for [0,1] scale

# LPIPS Plot (Normalized & Inverted)
sns.boxplot(ax=axes[2], data=df_melt_norm[df_melt_norm['metric'] == 'lpips'], x='approach', y='value_norm', palette="Reds") # <-- Use df_melt_norm, y='value_norm'
axes[2].set_title('Normalized LPIPS Distribution') # <-- Title updated
axes[2].set_xlabel('Approach')
axes[2].set_ylabel('Normalized LPIPS (Higher is Better)') # <-- Label updated (inverted)
axes[2].set_ylim(-0.05, 1.05) # Set Y axis limits for [0,1] scale

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_distribution_boxplot_normalized.png') # <-- Filename updated
print(f"Saved normalized distribution plot: {os.path.abspath('metrics_distribution_boxplot_normalized.png')}")
# plt.show()

# 2. Scatter Plots for Image-by-Image Comparison (NORMALIZED)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Image-wise Metric Comparison (Normalized Scale [0,1]): Approach 1 vs Approach 2', fontsize=16) # <-- Title updated

# Set fixed limits for normalized scale
norm_min, norm_max = -0.05, 1.05

# Helper line function (same as before)
def plot_identity(ax, min_val, max_val, **kwargs):
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, **kwargs)

# PSNR Scatter (Normalized)
sns.scatterplot(ax=axes[0], data=df, x='psnr_a1_norm', y='psnr_a2_norm', alpha=0.7) # <-- Use _norm columns
axes[0].set_title('Normalized PSNR: Approach 2 vs Approach 1') # <-- Title updated
axes[0].set_xlabel('Approach 1 Normalized PSNR') # <-- Label updated
axes[0].set_ylabel('Approach 2 Normalized PSNR') # <-- Label updated
axes[0].set_xlim(norm_min, norm_max) # Use fixed limits
axes[0].set_ylim(norm_min, norm_max) # Use fixed limits
plot_identity(axes[0], 0, 1, label='y=x line') # Plot identity line within [0,1]
axes[0].legend()
axes[0].grid(True)
axes[0].text(0.05, 0.95, 'Points above line favor Approach 2', transform=axes[0].transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

# SSIM Scatter (Normalized)
sns.scatterplot(ax=axes[1], data=df, x='ssim_a1_norm', y='ssim_a2_norm', alpha=0.7) # <-- Use _norm columns
axes[1].set_title('Normalized SSIM: Approach 2 vs Approach 1') # <-- Title updated
axes[1].set_xlabel('Approach 1 Normalized SSIM') # <-- Label updated
axes[1].set_ylabel('Approach 2 Normalized SSIM') # <-- Label updated
axes[1].set_xlim(norm_min, norm_max)
axes[1].set_ylim(norm_min, norm_max)
plot_identity(axes[1], 0, 1, label='y=x line')
axes[1].legend()
axes[1].grid(True)
axes[1].text(0.05, 0.95, 'Points above line favor Approach 2', transform=axes[1].transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

# LPIPS Scatter (Normalized & Inverted)
sns.scatterplot(ax=axes[2], data=df, x='lpips_a1_norm', y='lpips_a2_norm', alpha=0.7) # <-- Use _norm columns
axes[2].set_title('Normalized LPIPS: Approach 2 vs Approach 1') # <-- Title updated
axes[2].set_xlabel('Approach 1 Normalized LPIPS (Higher is Better)') # <-- Label updated
axes[2].set_ylabel('Approach 2 Normalized LPIPS (Higher is Better)') # <-- Label updated
axes[2].set_xlim(norm_min, norm_max)
axes[2].set_ylim(norm_min, norm_max)
plot_identity(axes[2], 0, 1, label='y=x line')
axes[2].legend()
axes[2].grid(True)
# Text updated: Because normalized LPIPS is inverted (higher is better), points ABOVE the line favor Approach 2
axes[2].text(0.05, 0.95, 'Points above line favor Approach 2', transform=axes[2].transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)) # <-- Text updated

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_scatter_comparison_normalized.png') # <-- Filename updated
print(f"Saved normalized scatter plot comparison: {os.path.abspath('metrics_scatter_comparison_normalized.png')}")
# plt.show()

# 3. Bar Chart of Average Metrics (NORMALIZED Scale)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Average Metric Comparison (Normalized Scale [0,1])', fontsize=16) # <-- Title updated

# Use the melted dataframe with NORMALIZED values
sns.barplot(ax=axes[0], data=df_melt_norm[df_melt_norm['metric'] == 'psnr'], x='approach', y='value_norm', palette="Blues", ci="sd") # <-- Use df_melt_norm, y='value_norm'
axes[0].set_title('Average Normalized PSNR') # <-- Title updated
axes[0].set_xlabel('')
axes[0].set_ylabel('Avg. Normalized PSNR') # <-- Label updated
axes[0].set_ylim(0, 1.0) # Set Y axis limits for [0,1] scale

sns.barplot(ax=axes[1], data=df_melt_norm[df_melt_norm['metric'] == 'ssim'], x='approach', y='value_norm', palette="Greens", ci="sd") # <-- Use df_melt_norm, y='value_norm'
axes[1].set_title('Average Normalized SSIM') # <-- Title updated
axes[1].set_xlabel('Approach')
axes[1].set_ylabel('Avg. Normalized SSIM') # <-- Label updated
axes[1].set_ylim(0, 1.0) # Set Y axis limits for [0,1] scale

sns.barplot(ax=axes[2], data=df_melt_norm[df_melt_norm['metric'] == 'lpips'], x='approach', y='value_norm', palette="Reds", ci="sd") # <-- Use df_melt_norm, y='value_norm'
axes[2].set_title('Average Normalized LPIPS') # <-- Title updated
axes[2].set_xlabel('')
axes[2].set_ylabel('Avg. Normalized LPIPS (Higher is Better)') # <-- Label updated
axes[2].set_ylim(0, 1.0) # Set Y axis limits for [0,1] scale

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('metrics_average_barplot_normalized.png') # <-- Filename updated
print(f"Saved normalized average metrics bar chart: {os.path.abspath('metrics_average_barplot_normalized.png')}")
# plt.show()

# --- Visualization Section End --- # <<< MODIFIED SECTION END >>>


# 4. Show Example Image Comparison (Showing Original Metrics - Unchanged)
def show_example(df, original_folder, a1_folder, a2_folder, index_to_show=0):
    if index_to_show >= len(df):
        print(f"Index {index_to_show} out of bounds ({len(df)} available). Showing first example.")
        index_to_show = 0
    if len(df) == 0:
         print("Cannot show example, DataFrame is empty.")
         return

    example_row = df.iloc[index_to_show]
    filename = example_row['filename']
    orig_file = os.path.join(original_folder, filename)
    a1_file = os.path.join(a1_folder, filename)
    a2_file = os.path.join(a2_folder, filename)

    if not all(os.path.exists(f) for f in [orig_file, a1_file, a2_file]):
        print(f"Skipping example for {filename}: One or more image files not found.")
        return

    try:
        img_orig = Image.open(orig_file)
        img_a1 = Image.open(a1_file)
        img_a2 = Image.open(a2_file)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6)) # Slightly taller figure
        fig.suptitle(f'Example Comparison: {filename}', fontsize=14)

        axes[0].imshow(img_orig)
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Display original metric values on the plot
        axes[1].imshow(img_a1)
        axes[1].set_title(f'Approach 1\n'
                          f'PSNR: {example_row.get("psnr_a1", "N/A"):.2f}\n'
                          f'SSIM: {example_row.get("ssim_a1", "N/A"):.3f}\n'
                          f'LPIPS: {example_row.get("lpips_a1", "N/A"):.3f}',
                          fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(img_a2)
        axes[2].set_title(f'Approach 2\n'
                          f'PSNR: {example_row.get("psnr_a2", "N/A"):.2f}\n'
                          f'SSIM: {example_row.get("ssim_a2", "N/A"):.3f}\n'
                          f'LPIPS: {example_row.get("lpips_a2", "N/A"):.3f}',
                          fontsize=10)
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        save_name = f'example_comparison_{os.path.splitext(filename)[0]}.png'
        plt.savefig(save_name)
        print(f"Saved example comparison image: {os.path.abspath(save_name)}")
        # plt.show()
    except Exception as e:
        print(f"Error generating example image for {filename}: {e}")


# Show the first image as an example
if not df.empty:
    show_example(df, ORIGINAL_FOLDER, APPROACH1_FOLDER, APPROACH2_FOLDER, index_to_show=0)
else:
    print("Skipping example image display due to empty results dataframe.")

print("\nProcessing Complete.")