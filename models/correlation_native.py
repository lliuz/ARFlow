import torch
import torch.nn as nn
import torch.nn.functional as F


class Correlation(nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = 2 * self.max_displacement + 1
        self.pad_size = self.max_displacement

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = F.pad(x2, [self.pad_size] * 4)
        cv = []
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)


if __name__ == '__main__':
    import time
    import random
    from correlation_package.correlation import Correlation as Correlation_cuda

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    corr1 = Correlation(max_displacement=4, kernel_size=1, stride1=1,
                            stride2=1, corr_multiply=1).to(device)

    corr2 = Correlation_cuda(pad_size=4, kernel_size=1, max_displacement=4, stride1=1,
                            stride2=1, corr_multiply=1)

    t1_sum = 0
    t2_sum = 0

    for i in range(50):
        C = random.choice([128, 256])
        H = random.choice([128, 256])  # , 512
        W = random.choice([64, 128])  # , 256
        x1 = torch.randn(4, C, H, W, requires_grad=True).to(device)
        x2 = torch.randn(4, C, H, W).to(device)

        end = time.time()
        y2 = corr2(x1, x2)
        t2_f = time.time() - end

        end = time.time()
        y2.sum().backward()
        t2_b = time.time() - end

        end = time.time()
        y1 = corr1(x1, x2)
        t1_f = time.time() - end

        end = time.time()
        y1.sum().backward()
        t1_b = time.time() - end

        assert torch.allclose(y1, y2, atol=1e-7)

        print('Forward: cuda: {:.3f}ms, pytorch: {:.3f}ms'.format(t1_f * 100, t2_f * 100))
        print(
            'Backward: cuda: {:.3f}ms, pytorch: {:.3f}ms'.format(t1_b * 100, t2_b * 100))

        if i < 3:
            continue
        t1_sum += t1_b + t1_f
        t2_sum += t2_b + t2_f

    print('cuda: {:.3f}s, pytorch: {:.3f}s'.format(t1_sum, t2_sum))
    ...

