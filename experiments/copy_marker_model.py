"""Source-format adapter preserving frozen content-target evaluators."""
import torch
from torch import nn

from copy_rope_model import make_model as base_model, reference as base_reference


def marked_source(content):
    if content.shape[1]==0:
        raise ValueError('Empty content source')
    return torch.cat((content,torch.full((len(content),1),2,dtype=torch.long,device=content.device)),1)


class MarkerModel(nn.Module):
    """The existing evaluator supplies content-length masks; extend only source masks."""
    def __init__(self, arm):
        super().__init__();self.base=base_model(arm)

    def features(self, content, prefix):
        return self.base.features(marked_source(content),prefix)

    def forward(self, source, target, sp, tp):
        if bool(sp.all(-1).any()):
            raise ValueError('Empty content source')
        if source.shape[1]!=sp.shape[1]+1:
            raise ValueError('Expected one appended source-end token')
        padding=torch.cat((sp,torch.zeros(len(sp),1,dtype=torch.bool,device=sp.device)),1)
        return self.base(source,target,padding,tp)


def make_model(arm):
    return MarkerModel(arm)


def reference(model, content, prefix, arm):
    # Independent formatting, rather than calling marked_source or adapter.features.
    source=torch.tensor([row+[2] for row in content.tolist()],dtype=torch.long)
    return base_reference(model.base,source,prefix,arm)
