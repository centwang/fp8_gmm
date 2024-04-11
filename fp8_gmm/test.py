import torch
import transformer_engine.pytorch as tex
from transformer_engine.common import recipe

from fp8_gmm import ops

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.glinear = ops.GroupedLinear(256, 128, 2, torch.bfloat16)

    def forward(self, input):
        return self.glinear(input, [128, 256])

input = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
model = TestModule()
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)
with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(input)
loss = out.sum()
loss.backward()
print(out)
print(input.grad)
