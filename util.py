import numpy as np
import torch
import torch.nn.functional as F


def calculate_ACC(pred, labels, threshold=0.5):
    pred = pred.detach().cpu().numpy().reshape(-1)
    one_hot_pred = np.zeros(pred.shape)
    one_hot_pred[pred > threshold] = 1
    labels = labels.detach().cpu().numpy().reshape(-1)
    return (one_hot_pred == labels).sum() / len(labels)

def MultiClassBCE(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if smoothing:
        ''' I Don't Actually Sure How to Do Label Smoothing Here '''
        raise RuntimeError('Do Your Label Smoothing Here')
        alpha = 0.2
        n_class = pred.size(1)

        # log_prb = F.logsigmoid(pred)
        # log_nprb = F.logsigmoid(-pred)
        # positive = gold * log_prb * ???
        # negative = (1-gold) * log_nprb * ???

        loss = -(positive + negative).sum(dim=1).mean()
    else:
        loss = F.binary_cross_entropy_with_logits(pred, gold, reduction='none')
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss, dim=0)
    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
