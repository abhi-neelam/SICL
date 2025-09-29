from math import e
import torch
import torch.nn.functional as F
from timm.optim.lars import Lars
from pytorch_lightning.core import LightningModule
from torch import nn
import numpy as np
from models.mlp import ProjectionMLP
from solo.losses.vicreg import vicreg_loss_func
from solo.losses.barlow import barlow_loss_func

class SICL(nn.Module):
    """SICL loss function modified from SupConLoss
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, beta=0.5, tau=0.2):
        super(SICL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.beta = beta
        self.tau = tau

    def forward(self, features, labels=None,super_labels=None, mask=None):
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


class NTXent(LightningModule):
    def __init__(self, batch_size, n_views=2, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.n_views, self.temperature)
        return self.criterion(logits, labels), pos, neg
    
    def get_infoNCE_logits_labels(self, features, batch_size, n_views=2, temperature=0.1):
        """
            Implementation from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        # creates a vector with labels [0, 1, 2, 0, 1, 2] 
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        # creates matrix where 1 is on the main diagonal and where indexes of the same intances match (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2) 
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
        similarity_matrix = get_cosine_sim_matrix(features)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.args.device)
        # mask out the main diagonal - output has one column less 
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # select and combine multiple positives
        positives = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives 
        negatives = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)

        # reshuffles values in each row so that positive similarity value for each row is in the first column
        logits = torch.cat([positives, negatives], dim=1)
        # labels is a zero vector because all positive logits are in the 0th column
        labels = torch.zeros(logits.shape[0])

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None):
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


class SimCLR_hcl(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, beta=0.5, tau=0.2):
        super(SimCLR_hcl, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.beta = beta
        self.tau = tau

    def forward(self, features, out_1, out_2 ):
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

        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        out = torch.cat([out_1, out_2], dim=0)


        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring

        N = batch_size * 2 - 2
        imp = (self.beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-self.tau * N * pos + reweight_neg) / (1 - self.tau)
            # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))


        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
  
        return loss


class SupCon_hcl(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,beta=0.5, tau=0.2):
        super(SupCon_hcl, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.beta = beta
        self.tau = tau

    def forward(self, features, out_1, out_2,indexes ):
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
        features = F.normalize(features, dim=2)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        out = torch.cat([out_1, out_2], dim=0)
        label = torch.cat([indexes, indexes], dim=0)
        cost = torch.exp(2 * torch.mm(out, out.t().contiguous()))
        batch = label.shape[0]
        pos_index = torch.zeros((batch, batch)).cuda()
        same_index = torch.eye(batch).cuda()
        for i in range(batch):
            ind = torch.where(label == label[i])[0]
            pos_index[i][ind] = 1
        neg_index = 1 - pos_index
        pos_index = pos_index - same_index
        pos = pos_index * cost
        neg = neg_index * cost
        imp = neg_index * (self.beta * neg.log()).exp()
        imp = imp.detach()
        neg_exp_sum = (imp * neg).sum(dim=-1) / imp.sum(dim=-1)
        Nce = pos_index * (pos / (pos + (batch - 2) * neg_exp_sum.reshape(-1, 1)))
        final_index = torch.where(pos_index != 0)
        Nce = (-torch.log(Nce[final_index[0], final_index[1]])).mean()
        return Nce


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def get_cosine_sim_matrix(features):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    return similarity_matrix


from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup phase
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay phase
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * 3.14159)))
            return cosine_decay

    return LambdaLR(optimizer, lr_lambda)



class SimCLRUnimodal_original(LightningModule):
    def __init__(self, modality, encoder, mlp_in_size, hidden=[256, 128], batch_size=64, temperature=0.1, n_views=2, optimizer_name_ssl='lars', lr=0.001, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters('modality', 'hidden', 'batch_size', 'temperature', 'n_views', 'optimizer_name_ssl', 'lr')
        self.encoder = encoder
        self.projection = ProjectionMLP(mlp_in_size, hidden)
        self.modality = modality

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.loss = NTXent(batch_size, n_views, temperature)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, pos, neg = self.loss(out)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = torch.cat(batch[self.modality], dim=0).float()
        out = self(batch)
        loss, _, _ = self.loss(out)
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

        elif self.optimizer_name_ssl.lower() == 'lars':
            optimizer = Lars(
                self.parameters(),
                self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                trust_coeff=0.001
            )

            return {
                "optimizer": optimizer
            }


class SimCLRUnimodal(LightningModule):
    def __init__(self, args, modality, encoder, mlp_in_size,framework='simclr', hidden=[256, 128], batch_size=64, temperature=0.1, n_views=2, optimizer_name_ssl='lars', lr=0.001, **kwargs) -> None:
        super().__init__()
        #self.save_hyperparameters()
        self.save_hyperparameters('modality', 'hidden', 'batch_size', 'temperature', 'n_views', 'optimizer_name_ssl', 'lr')

        self.encoder = encoder
        self.projection = ProjectionMLP(mlp_in_size, hidden)
        self.modality = modality
        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr
        self.temperature = args.temperature
        self.alpha=args.alpha

        self.loss = SupConLoss(temperature=args.temperature)
        self.framework = framework
        self.loss_sicl = SICL(temperature=args.temperature, beta=args.beta, tau=args.tau)
        self.loss_SimCLR_hcl = SimCLR_hcl(temperature=args.temperature,beta=args.beta,tau=args.tau)
        self.loss_SupCon_hcl = SupCon_hcl(temperature=args.temperature,beta=args.beta,tau=args.tau)

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = nn.Flatten()(x1)
        x2 = self.projection(x1)
        return x1 , x2

    def training_step(self, batch, batch_idx):
        x = torch.cat(batch[self.modality], dim=0).float()
        labels=batch['label'] 
        out1 , out2 = self(x)
        bsz=labels.size(0)
        f1, f2 = torch.split(out2, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # batch_size,2,128

        alpha=self.alpha

        if self.framework == 'simclr':
            loss=self.loss(features)
            #loss=simclr_loss_func(f,temperature=self.temperature)
        elif self.framework == 'SupCon':
            loss=self.loss(features,labels)
        elif self.framework == 'SI-SupCon':
            super_labels=batch['subject']
            loss=self.loss_sicl(features,labels,super_labels)
        elif self.framework == 'SICL':
            super_labels=batch['subject']
            loss=self.loss_sicl(features,super_labels=super_labels)
        elif self.framework == 'simclr_HCL':
            loss=self.loss_SimCLR_hcl(features,f1,f2)
        elif self.framework == 'supcon_HCL':
            loss=self.loss_SupCon_hcl(features,f1,f2,labels)
        elif(self.framework == 'vicreg'):
            loss = vicreg_loss_func(f1, f2)
        elif(self.framework == 'vicreg_SICL'):
            super_labels=batch['subject']
            loss=(1-alpha)*vicreg_loss_func(f1,f2)+alpha*self.loss_sicl(features,super_labels=super_labels)
        elif self.framework == 'barlow':
            loss=barlow_loss_func(f1,f2)
        elif self.framework == 'barlow_SICL':
            super_labels=batch['subject']
            loss=(1-alpha)*barlow_loss_func(f1,f2)+alpha*self.loss_sicl(features,super_labels=super_labels)        
        else:
            raise ValueError('Unknown framework: {}'.format(self.framework))
        #loss, pos, neg = self.loss(out)
        self.log("ssl_train_loss", loss)
        self.log("avg_positive_sim", loss)
        self.log("avg_neg_sim", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat(batch[self.modality], dim=0).float()
        labels=batch['label'] 
        out1, out2 = self(x)
        bsz=labels.size(0)
        f1, f2 = torch.split(out2, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # batch_size,2,128
        alpha=self.alpha
        if self.framework == 'simclr':
            loss=self.loss(features)
            #loss=simclr_loss_func(f,temperature=self.temperature)
        elif self.framework == 'SupCon':
            loss=self.loss(features,labels)
        elif self.framework == 'SI-SupCon':
            super_labels=batch['subject']
            loss=self.loss_sicl(features,labels,super_labels)
        elif self.framework == 'SICL':
            super_labels=batch['subject']
            loss=self.loss_sicl(features,super_labels=super_labels)
        elif self.framework == 'simclr_HCL':
            loss=self.loss_SimCLR_hcl(features,f1,f2)
        elif self.framework == 'supcon_HCL':
            loss=self.loss_SupCon_hcl(features,f1,f2,labels)
        elif(self.framework == 'vicreg'):
            loss = vicreg_loss_func(f1, f2)
        elif(self.framework == 'vicreg_SICL'):
            super_labels=batch['subject']
            loss=(1-alpha)*vicreg_loss_func(f1,f2)+alpha*self.loss_sicl(features,super_labels=super_labels)
        elif self.framework == 'barlow':
            loss=barlow_loss_func(f1,f2)
        elif self.framework == 'barlow_SICL':
            super_labels=batch['subject']
            loss=(1-alpha)*barlow_loss_func(f1,f2)+alpha*self.loss_sicl(features,super_labels=super_labels)        
        else:
            raise ValueError('Unknown framework: {}'.format(self.framework))

            
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

        elif self.optimizer_name_ssl.lower() == 'lars':
            optimizer = Lars(
                self.parameters(),
                self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                trust_coeff=0.001
            )
            
            warmup_epochs = 10  # Warm-up period
            max_epochs = 300  # Total epochs of training
            base_lr = self.lr
            min_lr_factor = 0.001  # Final learning rate is 1/1000 of the base learning rate
            
            scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, base_lr)
            
           
            return {
                "optimizer": optimizer
            }
