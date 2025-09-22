import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch import nn
import numpy as np
from models.mlp import ProjectionMLP

class MM_NTXent(LightningModule):
    """
    Multimodal adaptation of NTXent, according to the original CMC paper.
    """
    def __init__(self, batch_size, modalities, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.modalities = modalities
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.modalities, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix

    def get_infoNCE_logits_labels(self, features, batch_size, modalities=2, temperature=0.1):
        # Let M1 and M2 be abbreviations for the first and the second modality, respectively.

        # Computes similarity matrix by multiplication, shape: (batch_size, batch_size).
        # This computes the similarity between each sample in M1 with each sample in M2.
        features_1 = features[modalities[0]]
        features_2 = features[modalities[1]]
        similarity_matrix = MM_NTXent.get_cosine_sim_matrix(features_1, features_2)

        # We need to formulate (2 * batch_size) instance discrimination problems:
        # -> each instance from M1 with each instance from M2
        # -> each instance from M2 with each instance from M1

        # Similarities on the main diagonal are from positive pairs, and are the same in both directions.
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives_m1_m2 = similarity_matrix[mask].view(batch_size, -1)
        positives_m2_m1 = similarity_matrix[mask].view(batch_size, -1)
        positives = torch.cat([positives_m1_m2, positives_m2_m1], dim=0)

        # The rest of the similarities are from negative pairs. Row-wise for the loss from M1 to M2, and column-wise for the loss from M2 to M1.
        negatives_m1_m2 = similarity_matrix[~mask].view(batch_size, -1)
        negatives_m2_m1 = similarity_matrix.T[~mask].view(batch_size, -1)
        negatives = torch.cat([negatives_m1_m2, negatives_m2_m1])
        
        # Reshuffle the values in each row so that positive similarities are in the first column.
        logits = torch.cat([positives, negatives], dim=1)

        # Labels are a zero vector because all positive logits are in the 0th column.
        labels = torch.zeros(2 * batch_size)

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()




class SuperClass_CMC(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, beta=0.5, tau=0.2,modalities=2):
        super(SuperClass_CMC, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.beta = beta
        self.tau = tau
        self.modalities = modalities

    def forward(self, features,  labels=None,super_labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features_1 = features[self.modalities[0]]
        features_2 = features[self.modalities[1]]
        features = torch.cat([features_1.unsqueeze(1), features_2.unsqueeze(1)], dim=1) 
        features = F.normalize(features, dim=2)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            super_labels.contiguous().view(-1, 1)
            super_labels = super_labels.reshape(len(super_labels), 1)
            super_mask = torch.eq(super_labels, super_labels.T).float().to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            super_labels.contiguous().view(-1, 1)
            super_labels = super_labels.reshape(len(super_labels),1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
            super_mask = torch.eq(super_labels, super_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        super_mask = super_mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print where 1s are in logits_mask


        mask = mask * logits_mask

        super_mask =  logits_mask * super_mask
        neg_mask = logits_mask - super_mask

        # Super Negatives Only
        exp_logits_super = torch.exp(logits) * super_mask
        # Full Negatives without the SuperClass Negatives
        exp_logits = torch.exp(logits) * (neg_mask)
        # compute mean of log-likelihood over positive


        pos_matrix = torch.exp((mask * logits).sum())

        neg_log = torch.log(exp_logits_super.sum(1, keepdim=True))

        N = batch_size * 2 - 2
        imp = (self.beta * neg_log).exp()
        reweight_neg = (imp * neg_log).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-self.tau * N * pos_matrix + reweight_neg) / (1 - self.tau)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        log_prob = logits - Ng - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # check if there is NAN in loss
        if torch.isnan(loss.sum()):
            print("loss has NAN")
            raise ValueError('loss has NAN')
        return loss

class CMC(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, beta=0.5, tau=0.2,modalities=2):
        super(CMC, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.beta = beta
        self.tau = tau
        self.modalities = modalities

    def forward(self, features,  labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        features_1 = features[self.modalities[0]]
        features_2 = features[self.modalities[1]]

        features = torch.cat([features_1.unsqueeze(1), features_2.unsqueeze(1)], dim=1) 
        features = F.normalize(features, dim=2)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # check if there is NAN in anchor_dot_contrast

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #print("logits_max ", logits_max)
        logits = (anchor_dot_contrast - logits_max.detach())
        #print("logits ", logits)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # check if there is NAN in log_prob

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        # check if any zero in mask_pos_pairs

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # check if there is NAN in mean_log_prob_pos

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # check if there is NAN in loss
        if torch.isnan(loss.sum()):
            print("loss has NAN")
            raise ValueError('loss has NAN')
        return loss



class ContrastiveMultiviewCoding(LightningModule):
    """
    Implementation of CMC (contrastive multiview coding), currently supporting exactly 2 views.
    """
    def __init__(self, args, modalities, encoders, hidden=[256, 128], batch_size=64, temperature=0.1, optimizer_name_ssl='adam', lr=0.001, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'temperature', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

        self.projections = {}
        for m in modalities:
            self.projections[m] = ProjectionMLP(in_size=encoders[m].out_size, hidden=hidden)
        self.projections = nn.ModuleDict(self.projections)

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.loss_choice = args.loss
        self.temperature = args.temperature
        self.beta = args.beta
        self.tau = args.tau
        #self.loss = MM_NTXent(batch_size, modalities, temperature)
        self.loss = CMC(temperature=self.temperature, modalities=modalities)
        self.loss_super = SuperClass_CMC(temperature=self.temperature, modalities=modalities, beta=self.beta, tau=self.tau)
    def _forward_one_modality(self, modality, inputs):
        x = inputs[modality]
        x = self.encoders[modality](x)
        x = nn.Flatten()(x)
        x = self.projections[modality](x)
        return x

    def forward(self, x):
        outs = {}
        for m in self.modalities:
            outs[m] = self._forward_one_modality(m, x)
        return outs

    def training_step(self, batch, batch_idx):
        labels=batch['label'] 
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        #loss, pos, neg = self.loss(outs)
        if self.loss_choice == "super_subject":
            super_labels=batch['subject']
            loss = self.loss_super(outs,super_labels=super_labels)
        elif self.loss_choice == "simclr": 
            loss = self.loss(outs)
        else:
            raise ValueError("Loss choice not supported")
        self.log("ssl_train_loss", loss)
        #self.log("avg_positive_sim", pos)
        #self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        #loss, _, _ = self.loss(outs)
        #loss = self.loss(outs)
        super_labels=batch['subject']
        loss = self.loss_super(outs,super_labels=super_labels)
        self.log("ssl_val_loss", loss)

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }