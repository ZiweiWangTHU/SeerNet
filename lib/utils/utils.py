
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from lib.utils.quantize_utils import QConv2d, QLinear


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class MultiCropEnsemble(nn.Module):
    def __init__(self, module, cropsize, act=nn.functional.softmax, flipping=True):
        super(MultiCropEnsemble, self).__init__()
        self.cropsize = cropsize
        self.flipping = flipping
        self.internal_module = module
        self.act = act

    # Naive code
    def forward(self, x):
        # H, W >= cropsize
        assert(x.size()[2] >= self.cropsize)
        assert(x.size()[3] >= self.cropsize)

        cs = self.cropsize
        x1 = 0
        x2 = x.size()[2] - self.cropsize
        cx = x.size()[2] // 2 - self.cropsize // 2
        y1 = 0
        y2 = x.size()[3] - self.cropsize
        cy = x.size()[3] // 2 - self.cropsize // 2

        get_output = lambda x: self.act(self.internal_module.forward(x))

        _y = get_output(x[:, :, x1:x1+cs, y1:y1+cs])
        _y = get_output(x[:, :, x1:x1+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y1:y1+cs]) + _y
        _y = get_output(x[:, :, x2:x2+cs, y2:y2+cs]) + _y
        _y = get_output(x[:, :, cx:cx+cs, cy:cy+cs]) + _y

        if self.flipping == True:
            # Naive flipping

            arr = (x.data).cpu().numpy()                        # Bring back to cpu
            arr = arr[:,:,:, ::-1]                              # Flip
            x.data = type(x.data)(np.ascontiguousarray(arr))    # Store

            _y = get_output(x[:, :, x1:x1 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x1:x1 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y1:y1 + cs]) + _y
            _y = get_output(x[:, :, x2:x2 + cs, y2:y2 + cs]) + _y
            _y = get_output(x[:, :, cx:cx + cs, cy:cy + cs]) + _y

            _y = _y / 10.0
        else:
            _y = _y / 5.0

        return _y

class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if type(num)== str:
                self.file.write(num)
            else:
                self.file.write("{0:.6f}".format(num))


            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
from torch.autograd import Variable


def to_numpy(var):
    # return var.cpu().data.numpy()
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


