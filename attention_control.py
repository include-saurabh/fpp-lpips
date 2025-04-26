from typing import Union, Tuple
import torch
import abc


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + \
                        self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self,
                 num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = 2 # Assumes batch dim includes base and edited paths
        if isinstance(self_replace_steps, (int, float)): # Check if it's a single number
            # If 0, it means disable loss calculation. Set range to avoid execution.
            if self_replace_steps == 0:
                self_replace_steps = (num_steps + 1, num_steps) # Ensures condition is never met
            else:
                self_replace_steps = (0, self_replace_steps)
        elif not isinstance(self_replace_steps, tuple) or len(self_replace_steps) != 2:
             raise ValueError("self_replace_steps must be float or tuple of length 2")

        # Ensure start step is not negative and end step is reasonable
        start_step = max(0, int(num_steps * self_replace_steps[0]))
        end_step = max(start_step, int(num_steps * self_replace_steps[1]))
        self.num_self_replace = (start_step, end_step)
        self.loss = 0
        self.criterion = torch.nn.MSELoss() # Or L1Loss

        # Flag to check if attention loss should be calculated
        self.calculate_attn_loss = self.num_self_replace[0] < self.num_self_replace[1]


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # Store attention maps regardless of loss calculation (might be useful for other things)
        super(AttentionControlEdit, self).forward(
            attn, is_cross, place_in_unet)

        # Only proceed with loss calculation if enabled and within the step range
        if self.calculate_attn_loss and \
           not is_cross and \
           (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):

            # Ensure attention tensor has enough dimensions for batch splitting
            if attn.ndim >= 3: # Check shape, e.g., (batch*heads, seq_len, seq_len)
                 h = attn.shape[0] // self.batch_size # Calculate heads per batch item
                 if h > 0 and attn.shape[0] % self.batch_size == 0: # Ensure divisibility
                    try:
                        attn_reshaped = attn.reshape(self.batch_size, h, *attn.shape[1:])
                        attn_base, attn_repalce = attn_reshaped[0], attn_reshaped[1:]
                        # Accumulate loss ONLY IF conditions met and reshaping works
                        self.loss += self.criterion(
                            attn_repalce, self.replace_self_attention(attn_base, attn_repalce))

                    except RuntimeError as e:
                         print(f"Warning: Error reshaping attention tensor in {place_in_unet} at step {self.cur_step}. Shape: {attn.shape}, Batch size: {self.batch_size}. Error: {e}")
                         # Optionally handle the error, e.g., skip loss for this layer/step
                 else:
                      print(f"Warning: Cannot split attention batch dimension. Shape: {attn.shape}, Batch size: {self.batch_size} in {place_in_unet} at step {self.cur_step}")
            else:
                 print(f"Warning: Attention tensor has unexpected shape: {attn.shape} in {place_in_unet} at step {self.cur_step}")

        # Return the original attention map (modified in-place by super call if needed, but usually not)
        # Note: The reshaping above was only for loss calculation, not modifying the returned attn flow
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        # Expand base attention to match the shape of the replace attention batch dimension
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
