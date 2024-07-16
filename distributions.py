import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiDiscrete, MultiBinary


class Pd:
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return -self.neglogp(x)

    @property
    def shape(self):
        return self.flatparam().shape

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType:
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return torch.zeros(prepend_shape + self.param_shape())

    def sample_placeholder(self, prepend_shape, name=None):
        return torch.zeros(prepend_shape + self.sample_shape())


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    @property
    def mean(self):
        return F.softmax(self.logits, dim=-1)

    def neglogp(self, x):
        return F.cross_entropy(self.logits, x.long(), reduction='none')

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        a1 = other.logits - torch.max(other.logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        z1 = torch.sum(ea1, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=-1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)

    def sample(self):
        u = torch.rand(self.logits.size()).to(self.logits.device)
        return torch.argmax(self.logits - torch.log(-torch.log(u)), dim=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def _matching_fc(input, output_dim, init_scale=1.0, init_bias=0.0):
    layer = nn.Linear(input.size(-1), output_dim)
    nn.init.orthogonal_(layer.weight, gain=init_scale)
    nn.init.constant_(layer.bias, init_bias)
    return layer(input)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent_vector, self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return torch.int32


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = [CategoricalPd(logit) for logit in torch.split(flat, nvec, dim=-1)]

    def flatparam(self):
        return self.flat

    def mode(self):
        return torch.stack([p.mode() for p in self.categoricals], dim=-1)

    def neglogp(self, x):
        return sum(p.neglogp(px) for p, px in zip(self.categoricals, torch.unbind(x, dim=-1)))

    def kl(self, other):
        return sum(p.kl(q) for p, q in zip(self.categoricals, other.categoricals))

    def entropy(self):
        return sum(p.entropy() for p in self.categoricals)

    def sample(self):
        return torch.stack([p.sample() for p in self.categoricals], dim=-1)

    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec.astype('int32')
        assert (self.ncats > 0).all()

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def pdfromlatent(self, latent, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent, sum(self.ncats), init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return torch.int32


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        mean, logstd = torch.split(flat, flat.size(-1) // 2, dim=-1)
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)

    def flatparam(self):
        return torch.cat([self.mean, self.logstd], dim=-1)

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * torch.sum(torch.square((x - self.mean) / self.std), dim=-1) \
               + 0.5 * np.log(2.0 * np.pi) * x.size(-1) \
               + torch.sum(self.logstd, dim=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return torch.sum(other.logstd - self.logstd + (torch.square(self.std) + torch.square(self.mean - other.mean)) / (2.0 * torch.square(other.std)) - 0.5, dim=-1)

    def entropy(self):
        return torch.sum(self.logstd + 0.5 * np.log(2.0 * np.e), dim=-1)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        mean = _matching_fc(latent_vector, self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = nn.Parameter(torch.zeros(1, self.size))
        pdparam = torch.cat([mean, logstd.expand_as(mean)], dim=-1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.float32


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = torch.sigmoid(logits)

    def flatparam(self):
        return self.logits

    @property
    def mean(self):
        return self.ps

    def mode(self):
        return torch.round(self.ps)

    def neglogp(self, x):
        return F.binary_cross_entropy_with_logits(self.logits, x.float(), reduction='none').sum(dim=-1)

    def kl(self, other):
        return F.binary_cross_entropy_with_logits(other.logits, self.ps, reduction='none').sum(dim=-1) - \
               F.binary_cross_entropy_with_logits(self.logits, self.ps, reduction='none').sum(dim=-1)

    def entropy(self):
        return F.binary_cross_entropy_with_logits(self.logits, self.ps, reduction='none').sum(dim=-1)

    def sample(self):
        u = torch.rand_like(self.ps)
        return (u < self.ps).float()

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent_vector, self.size, init_scale=init_scale, init_bias=init_bias)
        return self.pd
