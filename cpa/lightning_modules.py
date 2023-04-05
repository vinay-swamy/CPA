
import pytorch_lightning as pl
import torch
import numpy as np  
from sklearn.metrics import r2_score

# class LitDataModule(pl.LightningDataModule):
#     def __init__(self, datasets, args, hparams):
#         datasets.update(
#             {
#                 "loader_tr": torch.utils.data.DataLoader(
#                     datasets["training"],
#                     batch_size=hparams["batch_size"],
#                     shuffle=True,
#                 )
#             }
#         )

        


class LitCPA(pl.LightningModule):
    def __init__(self,model,loss_config, optim_config, hparams):
        self.autoencoder = model
        self.loss_fns=self.configure_loss(loss_config)
        self.hparams = hparams
        self.optim_config = optim_config
    def training_step(self, batch, batch_idx):
        genes, perturbs, adv_class_labels,gene_target = batch
        gene_recon, latent_composition, adv_class_logits= self.autoencoder.predict(
            genes,
            perturbs
        )
        
        reconstruction_loss = self.loss_fns['reconstruction'](gene_recon, gene_target)
        
        adv_class_loss = self.loss_fns['adversarial'][0](adv_class_logits[0], adv_class_labels[0])
        for i in range(1,len(adv_class_logits)):
            adv_class_loss = adv_class_loss + self.loss_fns['adversarial'][i](adv_class_logits[i], adv_class_labels[i])
        
        self.log("train_loss_adv_class", adv_class_loss.item())

        # two place-holders for when adversary is not executed
        optimizer_adversaries, optimizer_autoencoder = self.optimizers()
        if self.global_step % self.hparams["adversary_steps"]:
            
            
            
            def compute_gradients(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            adv_class_penalty = compute_gradients(adv_class_logits.sum(), latent_composition) * self.hparams["penalty_adversary"]
            
            for i in range(1, len(adv_class_logits)):
                adv_class_penalty = adv_class_penalty + compute_gradients(adv_class_logits[i].sum(), latent_composition) * self.hparams["penalty_adversary"]

            
            
            self.log("train_loss_adv_penalty", adv_class_penalty.item())

            optimizer_adversaries.zero_grad()
            total_adv_loss = adv_class_loss + adv_class_penalty
            self.manual_backward(total_adv_loss)
            optimizer_adversaries.step()
            self.log("train_total_adv_loss", total_adv_loss)
        else:
            optimizer_autoencoder.zero_grad()
            total_recon_loss = reconstruction_loss - self.hparams["reg_adversary"] * adv_class_loss
            self.manual_backward(total_recon_loss)
            optimizer_autoencoder.step()
            self.log("train_total_recon_loss", total_recon_loss)
        
        return 
        
    def validation_step(self, batch, batch_idx):
        genes, perturbs, adv_class_labels,gene_target = batch
        gene_recon, latent_composition, adv_class_logits= self.autoencoder.predict(
            genes,
            perturbs
        )
        
        reconstruction_loss = self.loss_fns['reconstruction'](gene_recon, gene_target)
        
        adv_class_loss = self.loss_fns['adversarial'][0](adv_class_logits[0], adv_class_labels[0])
        for i in range(1,len(adv_class_logits)):
            adv_class_loss = adv_class_loss + self.loss_fns['adversarial'][i](adv_class_logits[i], adv_class_labels[i])
        ## will likely need to log `on_val_epoch_end`
        self.log("val_loss_adv_class", adv_class_loss.item())
        self.log("val_loss_recon", reconstruction_loss.item())
        r2_score = self.loss_fns['r2'](gene_recon, gene_target)
        self.log("val_r2_score", r2_score.item())
        return

    def configure_optimizers(self):
        # optimizers
        ## the recon optimizer optimizers the gene expressione encoder, the perturbation
        ## encoders, and the compositional decoder
        ## the adv optimizer optimizes the gene expression encoder and the adversarial classifiers
        optim_config = self.optim_config
        params_recon = list(self.autoencoder.gexp_encoder.parameters())
        for penc in self.autoencoder.perturb_encoders:
            params_recon.extend(list(penc.parameters()))
        
        params_recon.extend(list(self.autoencoder.compositional_decoder))
        
        optimizer_autoencoder = torch.optim.Adam(
            params_recon,
            lr=optim_config["recon"]['lr'],
            weight_decay=optim_config["recon"]['lr']
        )

        params_adv= list(self.autoencoder.gexp_encoder.parameters())
        for adv in self.autoencoder.adverserial_classifiers:
            params_adv.extend(list(adv.parameters()))

        optimizer_adversaries = torch.optim.Adam(
            params_adv,
            lr=optim_config["adversaries"]['lr'],
            weight_decay=optim_config["adversaries"]['lr']
        )

        ## lets not worry about schedulers for now 

        # learning rate schedulers
        # scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
        #     optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        # )

        # scheduler_adversary = torch.optim.lr_scheduler.StepLR(
        #     optimizer_adversaries, step_size=self.hparams["step_size_lr"]
        # )
        return optimizer_autoencoder, optimizer_adversaries
            
    def configure_loss(self,loss_config):
        loss_fns = {}
        lf_rc = loss_config['reconstruction']
        loss_fns['reconstruction'] = eval(lf_rc)()
        loss_fns['adversarial'] = []
        for lf_adv in loss_config['adversarial']:
            loss_fns['adversarial'].append(eval(lf_adv)())

        return 
        