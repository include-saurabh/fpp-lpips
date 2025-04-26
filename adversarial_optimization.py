# =============================================================================
# Import required libraries
# =============================================================================
import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import lpips

from criteria.cosine_loss import CosineLoss
from criteria.nce_loss import NCELoss
from attention_control import AttentionControlEdit
from utils import *

th_dict = {
    # Values for (FAR@0.1, FAR@0.01, FAR@0.001) - Keep consistent with tests.py
    'ir152': (0.094632, 0.166788, 0.227922),
    'irse50': (0.144840, 0.241045, 0.312703),
    'facenet': (0.256587, 0.409131, 0.591191),
    'mobile_face': (0.183635, 0.381611, 0.450878)
    # Add others if needed
}


@torch.enable_grad()
class Adversarial_Opt:
    def __init__(self, args, model):
        # ... (copy existing initializations) ...
        self.device = args.device
        self.dataloader = args.dataloader
        self.diff_model = model
        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)
        self.source_dir = args.source_dir
        self.protected_image_dir = args.protected_image_dir
        self.comparison_null_text = args.comparison_null_text
        self.target_choice = args.target_choice
        self.is_makeup = args.is_makeup
        self.source_text = args.source_text
        self.makeup_prompt = args.makeup_prompt
        self.MTCNN_cropping = args.MTCNN_cropping
        self.is_obfuscation = args.is_obfuscation
        self.image_size = args.image_size
        # Use max_prot_steps from args
        self.max_prot_steps = args.max_prot_steps
        self.diffusion_steps = args.diffusion_steps
        self.start_step = args.start_step
        self.null_optimization_steps = args.null_optimization_steps
        self.adv_optim_weight = args.adv_optim_weight
        self.makeup_weight = args.makeup_weight
        self.use_lpips_loss = args.use_lpips_loss
        self.lpips_weight = args.lpips_weight
        self.replace_attn_loss = args.replace_attn_loss
        self.augment = transforms.RandomPerspective(
            fill=0, p=1, distortion_scale=0.5)
        self.cosine_loss = CosineLoss(self.is_obfuscation)
        self.nce_loss = NCELoss(self.device, clip_model="ViT-B/32")
        if self.use_lpips_loss:
            print("Initializing LPIPS model...")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
            print("LPIPS model initialized.")
        self.surrogate_models = load_FR_models(
            args, args.surrogate_model_names)
        self.test_model_name = args.test_model_name[0] # Get the first (primary) test model name

        # --- New Initializations for Dynamic Stopping ---
        self.early_stop_success_criteria = args.early_stop_success_criteria
        self.early_stop_far_level = args.early_stop_far_level
        self.test_model_for_stop = None
        self.test_model_input_size = None
        self.early_stop_threshold = None
        self.reference_embedding_test_model = None # Will be calculated in run()

        if self.early_stop_success_criteria:
            print(f"Early stopping enabled. Checking against {self.test_model_name} at FAR@{self.early_stop_far_level}.")
            try:
                # Load the specific test model
                test_model_dict = load_FR_models(args, [self.test_model_name])
                self.test_model_input_size = test_model_dict[self.test_model_name][0]
                self.test_model_for_stop = test_model_dict[self.test_model_name][1]
                self.test_model_for_stop.eval() # Ensure it's in eval mode

                # Get the correct threshold index based on FAR level
                if self.early_stop_far_level == 0.1:
                    threshold_idx = 0
                elif self.early_stop_far_level == 0.01:
                    threshold_idx = 1
                elif self.early_stop_far_level == 0.001:
                    threshold_idx = 2
                else:
                    raise ValueError(f"Invalid early_stop_far_level: {self.early_stop_far_level}")

                self.early_stop_threshold = th_dict[self.test_model_name][threshold_idx]
                print(f"Using early stopping threshold: {self.early_stop_threshold:.6f}")

            except KeyError:
                print(f"ERROR: Test model '{self.test_model_name}' not found in load_FR_models or th_dict. Early stopping disabled.")
                self.early_stop_success_criteria = False
            except Exception as e:
                print(f"ERROR loading test model or threshold for early stopping: {e}. Early stopping disabled.")
                self.early_stop_success_criteria = False

    def get_FR_embeddings(self, image):
        features = []
        for model_name in self.surrogate_models.keys():
            input_size = self.surrogate_models[model_name][0]
            fr_model = self.surrogate_models[model_name][1]
            emb_source = fr_model(F.interpolate(
                image, size=input_size, mode='bilinear'))
            features.append(emb_source)
        return features

    def set_attention_control(self, controller):
        def ca_forward(self, place_in_unet):

            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                attn = controller(attn, is_cross, place_in_unet)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")

        controller.num_att_layers = cross_att_count

    def reset_attention_control(self):
        def ca_forward(self):
            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_)
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    register_recr(net__)

        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                register_recr(net[1])
            elif "up" in net[0]:
                register_recr(net[1])
            elif "mid" in net[0]:
                register_recr(net[1])

    def diffusion_step(self, latent, context, t, is_null_optimization=False, requires_grad=False, input_already_doubled=False):
        """
        Performs one step of the diffusion process. Handles gradients and dtype.
        """
        # Determine context based on requires_grad
        context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

        with context_manager:
            latent_dtype = latent.dtype # Preserve original latent dtype for output
            unet_dtype = self.diff_model.unet.dtype # Get expected UNet input dtype
            text_encoder_dtype = self.diff_model.text_encoder.dtype # Get expected context dtype

            # Ensure context is on the correct device and has the expected dtype
            context = context.to(device=latent.device, dtype=text_encoder_dtype)
            # Ensure latent has the expected UNet input dtype
            latent_unet_input = latent.to(dtype=unet_dtype)

            if is_null_optimization:
                # Single path for null-text optimization
                noise_pred = self.diff_model.unet(
                    latent_unet_input,
                    t,
                    encoder_hidden_states=context
                )["sample"]
                # Scheduler step uses single noise prediction and original latent shape
                prev_latent = self.diff_model.scheduler.step(noise_pred.to(latent_dtype), t, latent)["prev_sample"]

            elif input_already_doubled:
                # Input latent and context are already doubled [base, adv]
                # UNet processes both paths
                noise_pred_doubled = self.diff_model.unet(
                    latent_unet_input, # Shape [2, C, H, W]
                    t,
                    encoder_hidden_states=context # Shape [2, SeqLen, EmbDim]
                )["sample"]
                # Scheduler step applies to the doubled latent using the doubled noise prediction
                prev_latent = self.diff_model.scheduler.step(noise_pred_doubled.to(latent_dtype), t, latent)["prev_sample"]

            else:
                # Standard DDIM inversion or reconstruction: double internally
                latent_input = torch.cat([latent_unet_input] * 2)
                # Context must also be explicitly doubled to match latent batch size 2
                context_input = torch.cat([context] * 2)

                # No gradients usually needed here unless specifically required elsewhere
                with torch.no_grad():
                    noise_pred_doubled = self.diff_model.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=context_input
                    )["sample"]

                # Perform standard CFG logic (even if scale is 1, to get single noise pred)
                noise_pred_uncond, noise_pred_cond = noise_pred_doubled.chunk(2)
                # Since context is null repeated, uncond == cond.
                noise_pred = noise_pred_uncond # Or _cond, doesn't matter

                # Scheduler step uses the single effective noise prediction and the original latent
                prev_latent = self.diff_model.scheduler.step(noise_pred.to(latent_dtype), t, latent)["prev_sample"]

        # Return latent with the original input dtype
        return prev_latent.to(latent_dtype)

    def null_text_embeddings(self):
        uncond_input = self.diff_model.tokenizer([""],
                                                 padding="max_length",
                                                 max_length=self.diff_model.tokenizer.model_max_length,
                                                 return_tensors="pt")
        return self.diff_model.text_encoder(uncond_input.input_ids.to(self.device))[0]

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            generator = torch.Generator().manual_seed(8888)
            gpu_generator = torch.Generator(device=image.device)
            gpu_generator.manual_seed(generator.initial_seed())
            latents = self.diff_model.vae.encode(
                image).latent_dist.sample(generator=gpu_generator)
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latent):
        latent = 1 / 0.18215 * latent
        image = self.diff_model.vae.decode(latent)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def ddim_inversion(self, image):
        uncond_embeddings = self.null_text_embeddings()
        #
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)
        #
        latent = self.image2latent(image)
        all_latents = [latent]
        for i in tqdm(range(self.diffusion_steps - 1)):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]
            #
            noise_pred = self.diff_model.unet(latent,
                                              t,
                                              encoder_hidden_states=uncond_embeddings)["sample"]
            #
            next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
            alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep] \
                if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
            reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t]) * (
                latent - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
            latent = reverse_x0 * \
                torch.sqrt(alpha_bar_next) + \
                torch.sqrt(1 - alpha_bar_next) * noise_pred
            all_latents.append(latent)

        return all_latents

    def null_optimization(self, inversion_latents):
        """
        Optimizing the unconditional embeddings based on the paper:
            Null-text Inversion for Editing Real Images using Guided Diffusion Models
        GiHub:
            https://github.com/google/prompt-to-prompt
        """
        all_uncond_embs = []

        latent = inversion_latents[self.start_step - 1].clone()

        uncond_embeddings = self.null_text_embeddings()
        uncond_embeddings.requires_grad_(True)
        optimizer = optim.AdamW([uncond_embeddings], lr=1e-1) # Consider making LR an arg
        criterion = torch.nn.MSELoss()

        for i in tqdm(range(self.start_step, self.diffusion_steps), desc="Null-text optimization"):
            t = self.diff_model.scheduler.timesteps[i]
            current_target_latent = inversion_latents[i] # Target for this step

            # Inner optimization loop
            # Note: Optimizing uncond_embeddings based on matching the *next* inverted latent state
            for _ in range(self.null_optimization_steps):
                # Calculate the predicted previous state using current embeddings
                out_latent = self.diffusion_step(
                    latent,                # Latent from *previous* timestep (or start)
                    uncond_embeddings,     # Embeddings being optimized
                    t,
                    is_null_optimization=True,
                    requires_grad=True     # Need gradients for loss w.r.t embeddings
                )
                optimizer.zero_grad()
                # Ensure dtypes match for loss calculation
                loss = criterion(out_latent, current_target_latent.to(out_latent.dtype))
                loss.backward()
                optimizer.step()

            # Update latent state for the *next* outer loop iteration (i+1) using the *optimized* embeddings
            with torch.no_grad():
                latent = self.diffusion_step(
                    latent,                # Latent from *previous* timestep
                    uncond_embeddings,     # Use the final optimized embeddings for this step
                    t,
                    is_null_optimization=True,
                    requires_grad=False    # No gradients needed for state update
                ).detach()
                # Store the optimized embeddings for this timestep
                all_uncond_embs.append(uncond_embeddings.detach().clone())

        uncond_embeddings.requires_grad_(False)
        return all_uncond_embs

    def visualize(self, image_name, real_image, latents, controller):
     
        try: # Add try-except block (keep this from previous suggestion)
            adversarial_image = self.latent2image(latents)
            if adversarial_image is not None and len(adversarial_image) > 1:
                 protected_image_np = adversarial_image[1]
            else:
                 print(f"ERROR in visualize: latent2image did not return expected output for {image_name}")
                 return
        #adversarial_image = self.latent2image(latents)
        #adversarial_image = adversarial_image[1:]

        #result_dir = self.protected_image_dir + '/' + \
        #    self.test_model_name[0] + '/' + \
        #    self.target_choice + '/' + image_name
        
            base_result_dir = os.path.join(self.protected_image_dir, self.test_model_name, self.target_choice)
            output_filename = os.path.join(base_result_dir, f"{image_name}.png")
            # --- End Potential Issue Area ---


            print(f"Attempting to save image to: {output_filename}") # DEBUG PRINT

            adversarial_img_bgr = cv2.cvtColor(protected_image_np, cv2.COLOR_RGB2BGR)

            # Ensure the base directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True) # This should be base_result_dir

            save_success = cv2.imwrite(output_filename, adversarial_img_bgr)

            if save_success:
                print(f"Successfully saved {output_filename}") # DEBUG PRINT
            else:
                print(f"ERROR: cv2.imwrite failed to save {output_filename}") # DEBUG PRINT

        except Exception as e:
            print(f"ERROR during visualize for {image_name}: {e}")
            import traceback
            traceback.print_exc()

    def attacker(self,
                 image,
                 image_name,
                 source_embeddings,
                 target_embeddings,
                 controller,
                 reference_embedding_test_model,
                 null_text_dir=None,
                 bb_src1=None):
        # lat[0], lat[1], lat[2], ...
        inversion_latents = self.ddim_inversion(image)
        # reverse
        inversion_latents = inversion_latents[::-1]
        latent = inversion_latents[self.start_step - 1]
        #
        all_uncond_embs = self.null_optimization(inversion_latents)

        #######################################################################
        '''
        comparison between null_text and null_text optimized:

        '''
        if self.comparison_null_text:
            latent_holder = latent_holder_opt = latent.clone()
            uncond_embeddings = self.null_text_embeddings()
            #
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    #
                    latent_holder = self.diffusion_step(latent_holder,
                                                        uncond_embeddings,
                                                        t, True)
                    #
                    latent_holder_opt = self.diffusion_step(latent_holder_opt,
                                                            all_uncond_embs[i -
                                                                            self.start_step],
                                                            t, True)
                    #
                image_rec = self.latent2image(latent_holder)
                image_rec = cv2.cvtColor(image_rec[0], cv2.COLOR_RGB2BGR)
                result_dir = os.path.join(
                    null_text_dir, f"{image_name}_rec.png")
                cv2.imwrite(result_dir, image_rec)
                #
                image_rec_opt = self.latent2image(latent_holder_opt)
                image_rec_opt = cv2.cvtColor(
                    image_rec_opt[0], cv2.COLOR_RGB2BGR)
                result_dir = os.path.join(
                    null_text_dir, f"{image_name}_rec_opt.png")
                cv2.imwrite(result_dir, image_rec_opt)
                #
            return None
        #######################################################################
        reconstructed_image_for_lpips = None
        if self.use_lpips_loss:
            with torch.no_grad():
                latent_holder_rec = latent.clone()
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    latent_holder_rec = self.diffusion_step(latent_holder_rec,
                                                          all_uncond_embs[i - self.start_step],
                                                          t, True) # Use optimized null embeddings for reconstruction
                # Decode the reconstructed latent
                # VAE output is in [-1, 1], suitable for LPIPS
                reconstructed_image_for_lpips = self.diff_model.vae.decode(
                    1 / 0.18215 * latent_holder_rec)['sample']
                # Apply cropping consistent with FR embedding calculation if needed
                if self.MTCNN_cropping and bb_src1 is not None:
                     reconstructed_image_hold = reconstructed_image_for_lpips[:, :, round(bb_src1[1]):round(bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                     _, _, h, w = reconstructed_image_hold.shape
                     if h > 0 and w > 0: # Check if crop is valid
                         reconstructed_image_for_lpips = reconstructed_image_hold
                     else:
                         print(f"Warning: Invalid MTCNN crop for reconstruction {image_name}, using full image for LPIPS.")
                         # Fallback or handle error - For now, use uncropped if crop fails
                         reconstructed_image_for_lpips = self.diff_model.vae.decode(
                             1 / 0.18215 * latent_holder_rec)['sample']


        # --- End reconstruction generation --

        if self.is_makeup:
            latent_holder = latent.clone()
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    latent_holder = self.diffusion_step(latent_holder,
                                                        all_uncond_embs[i -
                                                                        self.start_step],
                                                        t, True)
            fast_render_image = self.diff_model.vae.decode(
                1 / 0.18215 * latent_holder)['sample']

        #
        self.set_attention_control(controller)
        
        num_guide_repeats = 2

        null_context_guidance = [[torch.cat([all_uncond_embs[i]] * 2)]
                        for i in range(len(all_uncond_embs))]
        null_context_guidance = [torch.cat(i) for i in null_context_guidance]


        init_latent = latent.clone()
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=1e-2) # Consider making LR an arg

        for step in tqdm(range(self.max_prot_steps), desc=f"Protecting {image_name}"):
            controller.loss = 0
            controller.reset()

            # --- Prepare Latents ---
            # Latent tensor for the diffusion step [base_recon_latent, adv_latent]
            latents_for_unet = torch.cat([init_latent, latent])

            # --- Diffusion process ---
            latents_ddpm = latents_for_unet.clone() # Operate on a clone within the DDIM loop
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                # Call diffusion_step with requires_grad=True and input_already_doubled=True
                latents_ddpm = self.diffusion_step(
                    latents_ddpm,                       # Input latent [Batch=2, C, H, W]
                    null_context_guidance[i - self.start_step], # Input context [Batch=2, Seq, Emb]
                    t,
                    requires_grad=True,                 # Enable gradients for optimization
                    input_already_doubled=True          # Signal that input is already doubled
                )
            # latents_ddpm now holds the result after diffusion [Batch=2, C, H, W]
            # --- End Diffusion ---

            # Decode the final latents
            # decoded_images will have shape [2, 3, H, W]
            decoded_images = self.diff_model.vae.decode(
                1 / 0.18215 * latents_ddpm.to(self.diff_model.vae.dtype) # Ensure VAE dtype match
                )['sample']

            # Extract the protected image (second half of the batch)
            protected_image = decoded_images[1:] # Shape [1, 3, H, W]
            protected_image_for_fr = protected_image
            protected_image_for_lpips = protected_image

            if self.MTCNN_cropping and bb_src1 is not None:
                protected_image_hold = protected_image[:, :, round(bb_src1[1]):round(bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                _, _, h, w = protected_image_hold.shape
                if h > 0 and w > 0: # Check if crop is valid
                    protected_image_for_fr = protected_image_hold
                    if self.use_lpips_loss: # Apply crop for LPIPS loss too
                        protected_image_for_lpips = protected_image_hold
                else:
                    print(f"Warning: Invalid MTCNN crop for protected image {image_name}, using full image.")
                    # Fallback or handle error
            # --- End Cropping ---
            
            # --- >>> Early Stopping Check <<< ---
            stop_early = False
            if self.early_stop_success_criteria and self.test_model_for_stop is not None and reference_embedding_test_model is not None:
                with torch.no_grad(): # No need for gradients during check
                    # Resize protected image for the test model
                    protected_img_resized = F.interpolate(
                        protected_image_for_fr, # Use the (potentially cropped) image
                        size=self.test_model_input_size,
                        mode='bilinear',
                        align_corners=False # Common practice
                    )
                    # Get embedding from the test model
                    current_embedding_test = self.test_model_for_stop(protected_img_resized)
                    # Calculate cosine similarity
                    cos_sim = F.cosine_similarity(current_embedding_test, reference_embedding_test_model).item()

                    # Check success criteria
                    if self.is_obfuscation:
                        # Obfuscation succeeds if similarity is LOW
                        if cos_sim < self.early_stop_threshold:
                            stop_early = True
                            print(f"\nINFO: Early stopping condition met for {image_name} (Obfuscation). Step: {step}, Similarity: {cos_sim:.4f} < Threshold: {self.early_stop_threshold:.4f}")
                    else:
                        # Impersonation succeeds if similarity is HIGH
                        if cos_sim > self.early_stop_threshold:
                            stop_early = True
                            print(f"\nINFO: Early stopping condition met for {image_name} (Impersonation). Step: {step}, Similarity: {cos_sim:.4f} > Threshold: {self.early_stop_threshold:.4f}")

            # --- End Early Stopping Check ---

            # --- Calculate Losses ---
            adv_loss = 0
            clip_loss = 0
            self_attn_loss = 0
            lpips_loss = 0
            loss = 0

            # 1. Adversarial Loss
            output_embeddings = self.get_FR_embeddings(protected_image_for_fr)
            adv_loss = self.cosine_loss(output_embeddings, target_embeddings, source_embeddings) * self.adv_optim_weight

            # 2. Makeup Loss
            # ... (makeup loss calculation) ...

            # 3. Structural Loss
            if self.use_lpips_loss:
                # Ensure reconstructed image is ready and has correct dtype/device
                rec_img_lpips = reconstructed_image_for_lpips.to(device=protected_image_for_lpips.device,
                                                                 dtype=protected_image_for_lpips.dtype)
                lpips_loss = self.lpips_model(protected_image_for_lpips, rec_img_lpips).mean()
                lpips_loss = lpips_loss * self.lpips_weight

                if not self.replace_attn_loss:
                    self_attn_loss = controller.loss # Additive attention loss
            else:
                self_attn_loss = controller.loss # Only attention loss

            # --- Total Loss ---
            loss = adv_loss + self_attn_loss + lpips_loss
            if self.is_makeup:
                loss += clip_loss

            # --- Logging ---
            print()
            print(f'Step {step}/{self.max_prot_steps}')
            print(f'  adv_loss: {adv_loss.item():.6f}')
            if not (self.use_lpips_loss and self.replace_attn_loss):
                 # Only print attn_loss if it's actually used
                 if torch.is_tensor(self_attn_loss):
                     print(f'  self_attn_loss: {self_attn_loss.item():.6f}')
                 else: # Should be 0 if not calculated
                     print(f'  self_attn_loss: {self_attn_loss}') # Print the integer value directly
            if self.use_lpips_loss:
                print(f'  lpips_loss: {lpips_loss.item():.6f} (weight: {self.lpips_weight})')
            if self.is_makeup:
                print(f'  clip_loss: {clip_loss.item():.6f}')
            print(f'  total_loss: {loss.item():.6f}')
            # --- End Logging ---

            # --- Optimization Step or Break ---
            if stop_early:
                # If condition met, break BEFORE optimizer step for this iteration
                break
            else:
                # If condition not met, proceed with optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # --- Final generation after optimization ---
        with torch.no_grad():
            controller.loss = 0
            controller.reset()

            # Prepare final latents [base, optimized_adv]
            latents_final_input = torch.cat([init_latent, latent.detach()]) # Detach optimized latent
            latents_final_output = latents_final_input.clone()

            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                # Use diffusion_step without gradients, input already doubled
                latents_final_output = self.diffusion_step(
                    latents_final_output,
                    null_context_guidance[i - self.start_step],
                    t,
                    requires_grad=False,
                    input_already_doubled=True
                )
        # latents_final_output contains [final_base_recon, final_protected]
        # --- End Final Generation ---

        self.reset_attention_control()
        # Return the final latents including the optimized one
        return latents_final_output.detach()

    def run(self):
        timer = MyTimer()
        time_list = []
        result_dir = os.path.join(self.protected_image_dir, self.test_model_name, self.target_choice)
        os.makedirs(result_dir, exist_ok=True) # Use exist_ok=True

        # --- Calculate Reference Embedding for Test Model ---
        self.reference_embedding_test_model = None
        if self.early_stop_success_criteria:
            with torch.no_grad():
                if self.is_obfuscation:
                    # For obfuscation, the reference is the SOURCE image's embedding
                    print("Calculating source embeddings for early stopping test model...")
                    # Load source images one by one or use the first one as reference?
                    # Let's assume we check against each source image dynamically
                    # This calculation needs to happen INSIDE the loop below
                    pass # We'll calculate per image inside the loop for obfuscation
                else:
                    # For impersonation, the reference is the TARGET image's embedding
                    print("Calculating target embedding for early stopping test model...")
                    target_image_stop, _ = get_target_test_images(
                        self.target_choice, self.device, self.MTCNN_cropping)
                    # Resize target image for the test model
                    target_img_resized = F.interpolate(
                        target_image_stop,
                        size=self.test_model_input_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    self.reference_embedding_test_model = self.test_model_for_stop(target_img_resized).detach()
                    print("Target embedding calculated.")
        # --- End Reference Embedding Calculation ---


        # Get target embeddings for surrogate models (loss calculation)
        target_image_surrogate, _ = get_target_test_images(
            self.target_choice, self.device, self.MTCNN_cropping)
        with torch.no_grad():
            target_embeddings_surrogate = self.get_FR_embeddings(target_image_surrogate)

        for i, (fname, image) in enumerate(self.dataloader):
            image_name = fname[0]
            image = image.to(self.device)
            print(f"\nProcessing image {i+1}/{len(self.dataloader)}: {image_name}")
            #
            bb_src1 = None
            if self.MTCNN_cropping:
                path = os.path.join(self.source_dir, image_name + '.png') # Use os.path.join
                try:
                    img = Image.open(path).convert('RGB') # Ensure RGB
                    if img.size[0] != self.image_size:
                        img = img.resize((self.image_size, self.image_size))
                    bb_src1 = alignment(img)
                    if bb_src1 is None: # Handle case where face is not detected
                         print(f"Warning: MTCNN failed to detect face for {image_name}. Cropping disabled for this image.")
                         # Optionally, skip this image or proceed without cropping
                         # bb_src1 = None # Ensure it's None if detection failed
                except FileNotFoundError:
                    print(f"Error: Source image not found at {path}")
                    continue # Skip this iteration
                except Exception as e:
                    print(f"Error processing {image_name} for MTCNN: {e}")
                    bb_src1 = None # Disable cropping on error


            # --- Initialize Attention Controller ---
            # If replacing attn loss, set self_replace_steps to 0 to disable attn loss calculation
            self_replace_steps_val = 1.0 # Default: calculate attn loss
            if self.use_lpips_loss and self.replace_attn_loss:
                 self_replace_steps_val = 0.0 # Disable attn loss calculation

            controller = AttentionControlEdit(num_steps=self.diffusion_steps,
                                              self_replace_steps=self_replace_steps_val)
            # --- End Controller Init ---

            if self.comparison_null_text:
                null_text_dir = os.path.join(
                    self.protected_image_dir, "null_text_opt")
                os.makedirs(null_text_dir, exist_ok=True)
            else:
                null_text_dir = None
            #
            source_embeddings_for_loss = None # Renamed to avoid conflict
            current_reference_embedding_test = self.reference_embedding_test_model # Use pre-calculated for impersonation
            if self.is_obfuscation:
                image_hold = image.clone()
                if self.MTCNN_cropping and bb_src1 is not None: # Check bb_src1 exists
                    out_image_hold = image_hold[:, :, round(bb_src1[1]):round(bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                    _, _, h, w = out_image_hold.shape
                    if h > 0 and w > 0: # Check crop validity
                        image_hold = out_image_hold
                    else:
                        print(f"Warning: Invalid MTCNN crop for source embedding calc {image_name}, using full image.")
                with torch.no_grad():
                    source_embeddings_for_loss = self.get_FR_embeddings(image_hold)

                    if self.early_stop_success_criteria and self.test_model_for_stop is not None:
                        source_img_resized = F.interpolate(
                            image_hold, # Use the potentially cropped source image
                            size=self.test_model_input_size,
                            mode='bilinear',
                            align_corners=False
                        )
                        current_reference_embedding_test = self.test_model_for_stop(source_img_resized).detach()

            # else: source_embeddings is None (already default)
            #
            timer.tic()
            #
            final_latents = self.attacker(image,
                                          image_name,
                                          source_embeddings_for_loss, # Pass potentially cropped embeddings
                                          target_embeddings_surrogate,
                                          controller,
                                          current_reference_embedding_test,
                                          null_text_dir,
                                          bb_src1) # Pass bb_src1 to attacker
            #
            avg_time = timer.toc()
            time_list.append(avg_time)

            if final_latents is not None:
                # Visualize uses the protected path latent (index 1)
                self.visualize(image_name, image, final_latents, controller)
        #
        print(f'Average protection time per image: {round(np.average(time_list), 2)} seconds') # More descriptive print
        result_fn = os.path.join(result_dir, "time.txt")
        # Use 'a' mode to append, ensure file exists or is created
        try:
            with open(result_fn, 'a') as f:
                f.write(f"Avg Time: {round(np.average(time_list),2)}\n")
        except IOError as e:
            print(f"Error writing time file: {e}")
