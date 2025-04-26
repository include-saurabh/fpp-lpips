

# Adversarial Diffusion for Facial Privacy Protection (Enhanced)

This repository implements and enhances a facial privacy protection technique using adversarial optimization within a Stable Diffusion framework, building upon the work from [parham1998/Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection). The goal is to generate protected versions of facial images that either:

1.  **Impersonate** a target identity, making Face Recognition (FR) models misclassify the protected image as the target.
2.  **Obfuscate** the original identity, making FR models fail to recognize the protected image as the original person.

## Enhancements Over Original Code

This version incorporates several key additions and modifications to the original codebase:

1.  **Perceptual Loss (LPIPS):** Integrated the Learned Perceptual Image Patch Similarity (LPIPS) loss as an optional structural guidance mechanism during the adversarial optimization (`--use_lpips_loss`, `--lpips_weight`). This aims to improve the visual quality and realism of the generated protected images by ensuring they remain perceptually similar to a reconstruction of the original image. This can be used to *augment* or *replace* (`--replace_attn_loss`) the original self-attention based structural loss.
2.  **Dynamic Early Stopping:** Implemented an optional early stopping criterion (`--early_stop_success_criteria`, `--early_stop_far_level`). This allows the optimization process for an individual image to terminate automatically once the generated image meets a pre-defined success threshold (cosine similarity corresponding to a specific FAR level) against the designated *test* FR model. This prevents over-optimization and can significantly reduce processing time per image.
3.  **Code Refinements:** Various fixes and improvements were made, including resolving tensor dimension mismatches during optimization, ensuring consistent path handling (`os.path.join`), adding robustness checks (e.g., for MTCNN failures, missing files), and improving debuggability with clearer print statements and error handling.
4.  **Argument Updates:** Renamed `--prot_steps` to `--max_prot_steps` when early stopping is enabled and added new arguments to control the LPIPS and early stopping features.

## Core Functionality (Inherited)

*   Based on Stable Diffusion v2-base.
*   DDIM Inversion for reconstructing initial latents.
*   Null-Text Inversion Optimization for improved image reconstruction fidelity.
*   Adversarial optimization targeting surrogate FR models (`Ir152`, `IRSE50`, `FaceNet`).
*   Evaluation against various test FR models (e.g., `MobileFaceNet`).
*   Optional MTCNN Cropping.
*   Optional experimental makeup transfer features.

## Setup

**1. Clone the Repository:**

```bash
# Replace 'your-username/your-repo-name.git' with the actual URL of your repository
git clone https://github.com/include-saurabh/fpp-lpips.git
cd your-repo-name
```

**2. Create Environment & Install Dependencies:**

(Using Conda is recommended)

```bash
# Create a new conda environment (Python 3.10 recommended)
conda create -n fpp python=3.10
conda activate fpp
# Install PyTorch matching your CUDA version.
# Go to https://pytorch.org/get-started/locally/ and select your preferences.
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
# Ensure requirements.txt includes: diffusers, transformers, accelerate, numpy, opencv-python, Pillow, tqdm, lpips, facenet-pytorch, scikit-image, clip-openai
```

**3. Prepare Assets Directory:**

Create an `assets/` directory in the project root and populate it as follows:

*   `assets/face_recognition_models/`: Place FR model weights (`.pth` files for `ir152`, `irse50`, `facenet`, `mobile_face`).
*   `assets/target_images/` & `assets/test_images/`: Place impersonation target images and corresponding test images.
*   `assets/obfs_target_images/`: Place obfuscation target image (if used).
*   `assets/datasets/`: Place source image datasets (e.g., `CelebA-HQ_aligned`).

## Usage

The main script for running the protection process is `main.py`.

**Example Command (Impersonation with LPIPS and Early Stopping @ FAR 0.001):**

```bash
python main.py \
    --source_dir assets/datasets/CelebA-HQ_aligned \
    --test_dir assets/datasets/CelebA-HQ_aligned \
    --MTCNN_cropping True \
    --target_choice 1 \
    --protected_image_dir results/impersonation_lpips_earlystop_example \
    --test_model_name mobile_face \
    --max_prot_steps 100 \
    --null_optimization_steps 20 \
    --adv_optim_weight 0.006 \
    --diffusion_steps 20 \
    --start_step 17 \
    --use_lpips_loss \
    --lpips_weight 0.5 \
    --early_stop_success_criteria \
    --early_stop_far_level 0.001
```

**Key Arguments (See `main.py` `parse_args` function for all options and defaults):**

*   `--source_dir`, `--test_dir`: Paths to the source images (to be protected) and corresponding original test images (if different).
*   `--MTCNN_cropping`: Set to `True` to enable MTCNN face detection and cropping before processing.
*   `--target_choice`: Index (1-based) of the target image in `assets/target_images/` for impersonation or `assets/obfs_target_images/` for obfuscation.
*   `--protected_image_dir`: Directory where the generated protected images will be saved.
*   `--test_model_name`: Name of the primary FR model used for evaluation and (if enabled) the early stopping criteria check (e.g., `mobile_face`).
*   `--max_prot_steps`: Maximum number of adversarial optimization steps per image. If early stopping is enabled, this acts as an upper limit.
*   `--adv_optim_weight`: Weight of the adversarial loss term.
*   `--use_lpips_loss`: Add LPIPS loss to the optimization objective.
*   `--lpips_weight`: Weight for the LPIPS loss term (requires tuning).
*   `--replace_attn_loss`: If `True`, LPIPS loss replaces the self-attention loss instead of augmenting it.
*   `--early_stop_success_criteria`: Enable dynamic early stopping based on the test model's performance.
*   `--early_stop_far_level`: False Accept Rate (FAR) level (e.g., `0.1`, `0.01`, `0.001`) corresponding to the cosine similarity threshold for successful protection (impersonation or obfuscation) used in early stopping.

## Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/f38c7096-8eb8-43c6-9b8b-7a21420c4df8/967f18aa-fc83-4ee3-a84f-0108936a3168" width="45%" />
  <img src="https://github.com/user-attachments/assets/252ed1e0-05e1-4a6d-9b10-f445da5618c8/4a46d6ac-cc91-4721-8075-4e7d4e847663" width="45%" />
</p>



## Acknowledgements

*   This code significantly builds upon and enhances the implementation found at [parham1998/Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection). Our sincere thanks to the original author.
*   This work leverages concepts, architectures, and tools from the following outstanding projects and libraries:
    *   [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
    *   [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
    *   [Null-Text Inversion](https://null-text-inversion.github.io/)
    *   [InsightFace](https://github.com/deepinsight/insightface) (for FR models like Ir152, IRSE50)
    *   [LPIPS (Learned Perceptual Image Patch Similarity)](https://github.com/richzhang/PerceptualSimilarity)
    *   [Facenet-PyTorch](https://github.com/timesler/facenet-pytorch)
    *   [CLIP (Contrastive Language–Image Pre-training)](https://github.com/openai/CLIP)
```
