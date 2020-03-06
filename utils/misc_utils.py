import collections


def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val
    return orig_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)
