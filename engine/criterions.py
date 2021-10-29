import torch 
import torch.nn.functional as F

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

def prob_kldiv(logits, prob_targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = prob_targets
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)

def cross_entropy(logits, targets, reduction='mean'):
    return F.cross_entropy(logits, targets, reduction=reduction)

def class_balance_loss(logits):
    prob = torch.softmax(logits, dim=1)
    avg_prob = prob.mean(dim=0)
    return (avg_prob * torch.log(avg_prob)).sum()

def onehot_loss(logits, targets=None):
    if targets is None:
        targets = logits.max(1)[1]
    return cross_entropy(logits, targets)

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    #loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def focal_loss(inputs, targets, alpha=1, gamma=0, size_average=True, ignore_index=255):
    ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    if size_average:
        return focal_loss.mean()
    else:
        return focal_loss.sum()