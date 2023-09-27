import torch
import torch.nn.functional as F
import torch.nn as nn

# component in (0, 1, 2)
class logit_loss(nn.Module):
    def __init__(self, confidence, labels):
        super().__init__()
        self.confidence = confidence
        self.labels = labels
        self.k = self.labels.sum(dim=1)
        self.sumprobs = 0.0

    def forward(self, outputs, index):
        curr_labels = self.labels[index, :]
        curr_k = self.k[index]

        probs = F.softmax(outputs, dim=1)
        sumprobs = (curr_labels * probs).sum(dim=1, keepdim=True)
        with torch.no_grad():
            pos_weights = (sumprobs - 1) * curr_labels
            neg_weights = (1 - sumprobs) * (1 - curr_labels)
            weights = pos_weights + neg_weights

        logits = outputs
        loss = (logits * weights).sum(dim=1).mean()
        self.sumprobs = sumprobs.mean()
        return loss

    def set_conf_ema_m(self, epoch, args):
        pass

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        pass

# loss = - log(sum probs)
# loss = sum_i (1/(sum probs) * p_i * (delta_ij - pj) * logit_j)
# these two should be equal
class nll_loss(nn.Module):
    def __init__(self, confidence, labels, on_logit=False):
        super().__init__()
        self.confidence = confidence
        self.labels = labels
        self.sumprobs = 0.0
        self.classes = labels.shape[1]
        self.delta = torch.unsqueeze(torch.eye(self.classes), 0)
        self.delta = self.delta.cuda()
        self.on_logit = on_logit

    def forward(self, outputs, index):
        curr_labels = self.labels[index, :]

        probs = F.softmax(outputs, dim=1)
        sumprobs = (curr_labels * probs).sum(dim=1, keepdim=True)
        
        if self.on_logit:
            with torch.no_grad():
                probs2 = torch.unsqueeze(probs, 2)
                # weight = probs2 * (self.delta - probs2)
                weight = probs2 * torch.transpose(self.delta - probs2, 1, 2)
                weight = weight * torch.unsqueeze(curr_labels, 2)
                weight = weight.sum(dim=1) / (sumprobs + 1e-8)
            loss = - (outputs * weight).sum(dim=1)
        else:
            loss = - torch.log(sumprobs)

        self.sumprobs = sumprobs.mean()
        loss = loss.mean()
        return loss

    def set_conf_ema_m(self, epoch, args):
        pass

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        pass

    
class prp_loss(nn.Module):
    def __init__(self, confidence, labels, component=0):
        super().__init__()
        self.confidence = confidence
        self.labels = labels
        self.k = self.labels.sum(dim=1)
        self.component = component
        self.log_threshold = torch.log(torch.tensor(1e-10))
        self.sumprobs = 0.0

    def forward(self, outputs, index):
        curr_labels = self.labels[index, :]
        curr_k = self.k[index]
        
        probs = F.softmax(outputs, dim=1) * curr_labels        
        logprobs = F.log_softmax(outputs, dim=1) * curr_labels
        logprobs = torch.maximum(logprobs, self.log_threshold)

        sumprobs = probs.sum(dim=1)
        prp1 = torch.log(1e-5 + 1-sumprobs)
        prp2 = logprobs.sum(dim=1) / curr_k
        with torch.no_grad():
            prp_weight = 1 - sumprobs
        prp = (prp1 - prp2) * prp_weight
        prp = torch.maximum(prp, torch.tensor(0.0))

        prp3 = - torch.log(1e-5 + sumprobs)
            
        if self.component == 0:            
            average_loss = prp.mean()
        elif self.component == 1:
            average_loss = prp1.mean()
        elif self.component == 2:
            average_loss = -prp2.mean()
        elif self.component == 3:
            average_loss = prp3.mean()

        self.sumprobs = sumprobs.mean()
        return average_loss

    def set_conf_ema_m(self, epoch, args):
        pass

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        pass

class prp_all_loss(nn.Module):
    def __init__(self, confidence, labels):
        super().__init__()
        self.confidence = confidence
        self.labels = labels
        self.k = self.labels.sum(dim=1)
        self.log_threshold = torch.log(torch.tensor(1e-5))
        self.sumprobs = 0.0

    def forward(self, outputs, index):
        curr_labels = self.labels[index, :]
        # curr_k = self.k[index]
        curr_k = curr_labels.sum(dim=1, keepdim=True)
        curr_k_neg = (1-curr_labels).sum(dim=1, keepdim=True)
        curr_k_neg = torch.maximum(curr_k_neg, torch.tensor(1.0))

        logprobs = F.log_softmax(outputs, dim=1)
        loss_allowed = - (curr_labels * logprobs).sum(dim=1)
        logprobs_disallowed = torch.maximum(torch.tensor(0.0), logprobs - self.log_threshold)
        loss_disallowed = ((1-curr_labels) * curr_k / curr_k_neg * logprobs_disallowed).sum(dim=1)

        loss = (loss_allowed + loss_disallowed).mean()

        probs = F.softmax(outputs, dim=1) * curr_labels        
        sumprobs = probs.sum(dim=1)
        self.sumprobs = sumprobs.mean()

        return loss
    
    def set_conf_ema_m(self, epoch, args):
        pass

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        pass


class partial_loss(nn.Module):
    def __init__(self, confidence, labels, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.labels = labels
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m
        self.sumprobs = 0.0

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()

        curr_labels = self.labels[index, :]
        probs = F.softmax(outputs, dim=1) * curr_labels        
        sumprobs = probs.sum(dim=1)
        self.sumprobs = sumprobs.mean()

        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
        return None

class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss
