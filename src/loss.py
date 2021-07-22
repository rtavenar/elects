import torch.nn.functional as F
import torch

def entropy(p):
    return -(p*torch.log(p+1e-12)).sum(1)

def build_t_index(batchsize, sequencelength):
    # linear increasing t index for time regularization
    """
    t_index
                          0 -> T
    tensor([[ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
    batch   [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            ...,
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.]])
    """
    t_index = torch.ones(batchsize, sequencelength) * torch.arange(sequencelength).type(torch.FloatTensor)
    if torch.cuda.is_available():
        return t_index.cuda()
    else:
        return t_index

def build_yhaty(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).type(torch.ByteTensor)
    if torch.cuda.is_available():
        eye = eye.cuda()

    # [b, t, c]
    targets_one_hot = eye[targets]

    # implement the y*\hat{y} part of the loss function
    y_haty = torch.masked_select(logprobabilities, targets_one_hot.bool())
    return y_haty.view(batchsize, seqquencelength).exp()


def early_loss_linear(logprobabilities, pts, targets, alpha=None, entropy_factor=0, ptsepsilon = 5):
    """
    Uses linear 1-P(actual class) loss. and the simple time regularization t/T
    L = (1-y\hat{y}) - t/T
    """
    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize,sequencelength=seqquencelength)

    ptsepsilon = ptsepsilon/seqquencelength

    y_haty = build_yhaty(logprobabilities, targets)

    loss_classification = alpha * ((pts+ptsepsilon) * (1 - y_haty)).sum(1).mean()
    loss_earliness = (1 - alpha) * ((pts+ptsepsilon) * (t_index / seqquencelength)).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()

    loss = loss_classification + loss_earliness + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_earliness=loss_earliness,
        loss_entropy=loss_entropy
    )

    return loss, stats

def early_loss_cross_entropy(logprobabilities, pts, targets, alpha=None, entropy_factor=0, ptsepsilon = 10):

    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize,sequencelength=seqquencelength)

    if ptsepsilon is not None:
        ptsepsilon = ptsepsilon / seqquencelength
        pts += ptsepsilon

    # reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
    loss_earliness = (1 - alpha) * (pts * (t_index / seqquencelength)).sum(1).mean()

    xentropy = F.nll_loss(logprobabilities.transpose(1,2).unsqueeze(-1), targets.unsqueeze(-1),reduction='none').squeeze(-1)
    loss_classification = alpha * (pts*xentropy).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()

    loss = loss_classification + loss_earliness + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_earliness=loss_earliness,
    )

    return loss, stats

def loss_cross_entropy(logprobabilities, pts, targets,weight=None):

    b,t,c = logprobabilities.shape
    loss = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t), ignore_index=-1, weight=weight)

    stats = dict(
        loss=loss,
    )

    return loss, stats


def loss_cross_entropy_entropy_regularized(logprobabilities,pts, targets, entropy_factor=0.1):

    b,t,c = logprobabilities.shape
    #loss_classification = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))
    xentropy = F.nll_loss(logprobabilities.transpose(1, 2).unsqueeze(-1), targets.unsqueeze(-1),
                          reduction='none').squeeze(-1)
    loss_classification = (pts * xentropy).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()
    loss = loss_classification + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_entropy=loss_entropy
    )

    return loss, stats

def loss_early_reward(logprobabilities,pts, targets, alpha=1, ptsepsilon = 10, power=1):

    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize, sequencelength=seqquencelength)

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    if ptsepsilon is not None:
        ptsepsilon = ptsepsilon / seqquencelength
        #pts += ptsepsilon

    #pts_ = torch.nn.functional.softmax((pts + ptsepsilon), dim=1)
    pts_ = (pts + ptsepsilon)

    b,t,c = logprobabilities.shape
    #loss_classification = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))
    xentropy = F.nll_loss(logprobabilities.transpose(1, 2).unsqueeze(-1), targets.unsqueeze(-1),
                          reduction='none').squeeze(-1)
    loss_classification = alpha * ((pts_ * xentropy)).sum(1).mean()

    yyhat = build_yhaty(logprobabilities, targets)
    earliness_reward = (1-alpha) * ((pts) * (yyhat)**power * (1 - (t_index / seqquencelength))).sum(1).mean()

    loss = loss_classification - earliness_reward

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        earliness_reward=earliness_reward
    )

    return loss, stats
