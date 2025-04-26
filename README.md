```markdown
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
*   Adversarial optimization targeting surrogate FR models (Ir152, IRSE50, FaceNet).
*   Evaluation against various test FR models (e.g., MobileFaceNet).
*   Optional MTCNN Cropping.
*   Optional experimental makeup transfer features.

## Setup

**1. Clone the Repository:**

```bash
git clone https://github.com/your-username/your-repo-name.git # Replace with your repo URL
cd your-repo-name
```

**2. Create Environment & Install Dependencies:**

(Using Conda is recommended)

```bash
conda create -n fpp python=3.10
conda activate fpp
# Install PyTorch matching your CUDA version (see PyTorch website)
# e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Ensure requirements.txt includes: diffusers, transformers, accelerate, numpy, opencv-python, Pillow, tqdm, lpips, facenet-pytorch, scikit-image, clip-openai
```

**3. Prepare Assets Directory:**

Create an `assets/` directory and populate it:

*   `assets/face_recognition_models/`: Place FR model weights (`.pth` files for ir152, irse50, facenet, mobile_face).
*   `assets/target_images/` & `assets/test_images/`: Place impersonation target images and corresponding test images.
*   `assets/obfs_target_images/`: Place obfuscation target image (if used).
*   `assets/datasets/`: Place source image datasets (e.g., `CelebA-HQ_aligned`).

## Usage

The main script is `main.py`.

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
    --adv_optim_weight 0.003 \
    --diffusion_steps 20 \
    --start_step 17 \
    --use_lpips_loss \
    --lpips_weight 0.5 \
    --early_stop_success_criteria \
    --early_stop_far_level 0.001
```

**Key Arguments (See `main.py` `parse_args` for details):**

*   `--source_dir`, `--test_dir`: Dataset paths.
*   `--MTCNN_cropping`: Enable face cropping.
*   `--target_choice`: Impersonation/Obfuscation target ID.
*   `--protected_image_dir`: Output directory.
*   `--test_model_name`: Primary FR model for evaluation and early stopping check.
*   `--max_prot_steps`: Max optimization steps (used if early stopping is off or not met).
*   `--adv_optim_weight`: Adversarial loss weight.
*   `--use_lpips_loss`: Enable LPIPS loss.
*   `--lpips_weight`: LPIPS loss weight (tune this).
*   `--replace_attn_loss`: Use LPIPS *instead* of attention loss.
*   `--early_stop_success_criteria`: Enable dynamic stopping.
*   `--early_stop_far_level`: FAR threshold (0.1, 0.01, 0.001) for early stopping.

**Running Evaluation:**

Modify `main.py` to skip the generation (`adversarial_opt.run()`) and only run the evaluation functions (`attack_local_models` / `attack_local_models_obfuscation`). Run `main.py` pointing `--protected_image_dir` to the results you want to evaluate. Use the modified `tests.py` / `test_obfs.py` if you need to report specific FAR levels only.

## File Structure

```
.
├── main.py                 # Main script
├── adversarial_optimization.py # Core optimization logic (enhanced)
├── attention_control.py    # Attention manipulation
├── criteria/               # Loss functions
├── dataset.py              # Dataset loader
├── tests.py                # Impersonation evaluation
├── test_obfs.py            # Obfuscation evaluation
├── utils.py                # Helpers (model loading etc.)
├── assets/                 # Data and models (User needs to populate)
├── results/                # Default output directory
└── requirements.txt        # Dependencies
```

## Acknowledgements

*   This code significantly builds upon and enhances the implementation found at [parham1998/Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection).
*   Leverages concepts and tools from Stable Diffusion, Diffusers, Null-Text Inversion, InsightFace, LPIPS, Facenet-PyTorch, and CLIP.
```
