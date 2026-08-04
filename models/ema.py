import copy

import torch.nn as nn

from utils.printing import rank_zero_print_warning


class EMAHelper:
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        previous = {name.replace("_orig_mod.", ""): value for name, value in self.shadow.items()}
        self.shadow = {}
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                value = previous.get(name.replace("_orig_mod.", ""))
                self.shadow[name] = param.data.clone() if value is None else value

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    rank_zero_print_warning(f"EMAHelper: registering new parameter '{name}' for EMA tracking.")
                    self.shadow[name] = param.data.clone()
                    continue
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                v = self.shadow.get(name)
                if v is None:
                    rank_zero_print_warning(f"EMAHelper: parameter '{name}' not found in EMA shadow; skipping EMA copy.")
                    continue
                param.data.copy_(v.data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
