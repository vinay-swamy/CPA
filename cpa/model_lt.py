# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from http.client import RemoteDisconnected
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, mu, y, theta, eps=1e-8):
        """Negative binomial negative log-likelihood. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + y * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1)
        )
        res = _nan2inf(res)
        return -torch.mean(res)
    
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm, last_layer_act):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)

class DecoderMLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm, last_layer_act):
        super(DecoderMLP, self).__init__()
        self.mlp = MLP(sizes, batch_norm, last_layer_act)
    def forward(self, x):
        x_mu_sigma = self.mlp(x)
        dim = x_mu_sigma.size(1) // 2
        gene_means = x_mu_sigma[:, :dim]
        gene_vars = F.softplus(x_mu_sigma[:, dim:]).add(1e-3)
        return gene_means, gene_vars
        


class GeneralizedSigmoid(torch.nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, device, nonlin="sigmoid"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim), requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim), requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == "logsigm":
            c0 = self.bias.sigmoid()
            return (torch.log1p(x) * self.beta + self.bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias.sigmoid()
            return (x * self.beta + self.bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.nonlin == "logsigm":
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.nonlin == "sigm":
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class EmbeddingPerturbationEncoder(torch.nn.Module):
    def __init__(self, num_perturbations, dim,):
        super(EmbeddingPerturbationEncoder, self).__init__()
        self.num_perturbations = num_perturbations
        self.dim = dim
        self.embedding = torch.nn.Embedding(
            self.num_perturbations, self.dim
        )
    def forward(self, x):
        return self.embedding(x)

class DrugPerturbationEncoder(torch.nn.Module):
    def __init__(self, dose_encoder, perturbation_encoder):
        ## follow notation of chemCPA paper 
        super(DrugPerturbationEncoder, self).__init__()
        _dose_encoder = eval(dose_encoder['fn'])
        self.dose_encoder =  _dose_encoder(**dose_encoder['args'])
        _final_perturbation_encoder = eval(perturbation_encoder['fn'])
        self.final_perturbation_encoder = _final_perturbation_encoder(**perturbation_encoder['args'])
    def forward(self,  batch):
        mol_enc, dose = batch
        dose_enc = self.dose_encoder(torch.concat([mol_enc, dose.unsqueeze(-1)], dim=1) )
        latent = self.final_perturbation_encoder(mol_enc * dose_enc)
        return latent 



class ModularCPA(torch.nn.Module):
    """ 
    Module CPA is composed of several components:
    - One Gene Expression Encoder
    - One or more perturbation encoders 
    - one or more adversaial classifers 
    - one latent decoder
    """
    def __init__(
        self,
        num_genes,
        num_perturbations,
        perturbation_enc_dims,
        num_covariates,
        gene_expression_encoder,
        perturbation_encoders,
        adversarial_classifiers,
        latent_decoder,
        
    ):
        super(ModularCPA, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_pertubations = num_perturbations
        self.num_covariates = num_covariates


        # set models
        _gexp_encoder = eval(gene_expression_encoder['fn'])
        self.gexp_encoder = _gexp_encoder(**gene_expression_encoder['args'])

        _compostional_decoder = eval(latent_decoder['fn'])
        self.compositional_decoder = _compostional_decoder(**latent_decoder['args'])
        

        self.adverserial_classifiers = torch.nn.ModuleList()
        for adv_cfg in adversarial_classifiers:
            _model = eval(adv_cfg['fn'])
            model = _model(**adv_cfg['args'])
            self.adverserial_classifiers.append(model)
        self.perturbation_encoders = torch.nn.ModuleList()
        for penc in perturbation_encoders:
            _model = eval(penc['fn'])
            model = _model(**penc['args'])
            self.perturbation_encoders.append(model)

    def forward(
        self, 
        gexp, 
        perturbations
    ):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with cell types `cell_types` been treated with
        combination of drugs `drugs`.
        """

        latent_z = self.gexp_encoder(gexp)
        latent_compostion= latent_z
        for i in range(len(perturbations)):
            latent_compostion = latent_compostion + self.perturbation_encoders[i](perturbations[i])    

        adverserial_classifications = [self.adverserial_classifiers[i](latent_compostion) for i in range(len(self.adverserial_classifiers))]

        

        gene_reconstructions = self.compositional_decoder(latent_compostion)
            # convert variance estimates to a positive value in [1e-3, \infty)
        return gene_reconstructions, latent_z, adverserial_classifications
        

