import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from torch import _weight_norm

    

class LipWeightNorm(WeightNorm):
    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = LipWeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_c', nn.Parameter(torch.tensor(1.)))
        module.register_parameter(name + '_v', nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn
    
    def compute_weight(self, module):
        w = getattr(module, self.name + '_v')
        c = getattr(module, self.name + '_c')
        absrowsum = torch.sum(torch.abs(w), 1, keepdim=True)
        scale = (c / absrowsum).clamp(min=1)  # min(1, c / abs)
        # recompute weight as w * scale
        return _weight_norm(w, scale, self.dim)
        
        
def lip_weight_norm(module, name: str = 'weight', dim: int = 0):
    LipWeightNorm.apply(module, name, dim)
    return module
