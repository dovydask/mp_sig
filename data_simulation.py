import numpy as np

# Adapted from code by nikulam1 circa 2017


def random_mu(N_signatures, N_channels, mu_dirichlet_alpha_min, mu_dirichlet_alpha_max):
    alphas = np.linspace(mu_dirichlet_alpha_min, mu_dirichlet_alpha_max, num=N_signatures)
    mu = np.zeros((N_signatures, N_channels))
    for j in range(N_signatures):
        mu[j] = np.random.dirichlet(alphas[j]*np.ones(N_channels))

    return mu


def random_x(N_samples, N_signatures, N_active_signatures, x_lognormal_scale):
    x = np.zeros((N_samples, N_signatures))
    for i in range(N_samples):
        active_signatures = np.random.choice(N_signatures, size=N_active_signatures, replace=False)
        activities = np.exp(np.random.normal(loc=0, scale=x_lognormal_scale, size=N_active_signatures))
        for j in range(N_active_signatures):
            x[i, active_signatures[j]] = activities[j]

    return x


def generate_data_mul_process(c, mp, N_samples=1000, N_signatures=15, N_channels=96,
                              mu_dirichlet_alpha_min=0.5, mu_dirichlet_alpha_max=2,
                              N_active_signatures=5, x_lognormal_scale=1.5, multiplier=50,
                              data_path="../data/", filename="a",
                              signatures=None, manual_signatures=False):

    mu = random_mu(N_signatures, N_channels, mu_dirichlet_alpha_min, mu_dirichlet_alpha_max)
    x_pre = random_x(N_samples, N_signatures, N_active_signatures, x_lognormal_scale)

    if manual_signatures:
        print("Signatures given manually")
        mu = signatures
        if N_signatures == 1:
            mu = mu.reshape(1, 96)

    m = np.dot(x_pre, mu)
    x2 = multiplier*x_pre/np.mean(m)

    p = 1 + c.reshape(-1, 1)*mp.reshape(1, -1)
    lam = multiplier*p*m/np.mean(p*m)

    data = np.random.poisson(lam)
    x = multiplier*x_pre/np.mean(p*m)
    
    np.savetxt(data_path + "{}-x2.txt".format(filename), x2, fmt='%5.1i')
    np.savetxt(data_path + "{}-x_pre.txt".format(filename), x_pre, fmt='%5.1i')
    np.savetxt(data_path + "{}-samples.txt".format(filename), data, fmt='%5.1i')
    np.savetxt(data_path + "{}-mu.txt".format(filename), mu, fmt='%.5f')
    np.savetxt(data_path + "{}-x.txt".format(filename), x, fmt='%8.2f')

    return None
