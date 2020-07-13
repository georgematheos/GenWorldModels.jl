import math
import torch
from torch.nn import Parameter
from torch.optim import Adam
from torch.distributions import Gamma, Normal, LogNormal

_alpha_0 = Parameter(torch.zeros(1))
_beta_0 = Parameter(torch.zeros(1))
_kappa_0 = Parameter(torch.zeros(1))
mu_0 = Parameter(torch.zeros(1))

durations = torch.Tensor([
    [0.06, 0.07, 0.08],
    [0.45, 0.5, 0.55],
    [0.07, 0.5, 1.0],
    [0.07, 0.1, 0.2],
])
log_durations = durations.log()
x = log_durations

optimizer = Adam([_alpha_0, _beta_0, _kappa_0, mu_0], lr=1e-2)
for i in range(2000):
    optimizer.zero_grad()
    alpha_0, beta_0, kappa_0 = _alpha_0.exp(), _beta_0.exp(), _kappa_0.exp()

    # Murphy, Eq, (87-89)
    # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    n = x.shape[-1]
    alpha_n = alpha_0 + n/2
    x_bar = x.mean(axis=-1)
    beta_n = beta_0 + \
        0.5 * ((x - x_bar[:, None])**2).sum(axis=-1) + \
        kappa_0 * n * (x_bar - mu_0)**2 / (2 * (kappa_0 + n))
    kappa_n = kappa_0 + n

    # Murphy, Eq. (95)
    # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    score_per_observation = \
        alpha_n.lgamma() - alpha_0.lgamma() + \
        alpha_0 * beta_0.log() - alpha_n * beta_n.log() + \
        0.5 * (kappa_0.log() - kappa_n.log()) + \
        -n/2 * math.log(2*math.pi)
    score = score_per_observation.sum()

    (-score).backward()
    optimizer.step()
    if i % 100 == 0:
        print("Iteration {} score {:.2f} | a_0={:.2f} b_0={:.2f} k_0={:.2f} m_0={:.2f}".format(
            i, score.item(), alpha_0.item(), beta_0.item(), kappa_0.item(), mu_0.item()))

print("Samples:")
for _ in range(50):
    precision = Gamma(alpha_0, beta_0).sample()
    mean = Normal(mu_0, 1/(precision * kappa_0).sqrt()).sample()
    x = LogNormal(mean, 1/precision.sqrt()).sample(sample_shape=[3])
    print(x[:, 0].tolist())