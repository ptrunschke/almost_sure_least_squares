import numpy as np
from tqdm import tqdm, trange

from least_squares import mu_W
# from sampling import Sampler
from sampling_new import Sampler


def greedy_subsample(sample, sample_size, sampler):
    while len(sample) > sample_size:
        mus = []
        for idx in range(len(sample)):
            sample_without_idx = sample[:idx] + sample[idx + 1 :]
            mus.append(mu_W(sample_without_idx, sampler.dimension, sampler.basis))
        idx = np.argmin(mus)
        sample = sample[:idx] + sample[idx + 1 :]
    return sample

def greedy_swap(sample, candidates):
    mus = []
    for idx in range(len(candidates)):
        sample_with_idx = sample + [candidates[idx]]
        mus.append(mu_W(sample_with_idx, sampler.dimension, sampler.basis))
    idx = np.argmin(mus)
    sample = sample + [candidates[idx]]
    mus = []
    for idx in range(len(sample)):
        sample_without_idx = sample[:idx] + sample[idx + 1 :]
        mus.append(mu_W(sample_without_idx, sampler.dimension, sampler.basis))
    idx = np.argmin(mus)
    sample = sample[:idx] + sample[idx + 1 :]
    return sample

def greedy_optimise(sample_or_size, sampler, rng, max_iterations=100, tolerance=1e-1):
    if isinstance(sample_or_size, int):
        sample = []
        while len(sample) < sample_size:
            sample.extend(sampler.draw(rng))
        sample = sample[:sample_size]
    else:
        sample = sample_or_size
    mu_W_new = mu_W(sample, sampler.dimension, sampler.basis)
    for _ in range(max_iterations):
        mu_W_old = mu_W_new
        sample = greedy_swap(sample, sampler.draw(rng))
        mu_W_new = mu_W(sample, sampler.dimension, sampler.basis)
        assert mu_W_old >= mu_W_new
        if mu_W_old - mu_W_new < tolerance:
            break
    return sample, mu_W_new

def greedy_optimise_mu(mu_bound, sampler, rng, verbose=False):
    if verbose:
        print("Create sample")
    sample = list(sampler.draw(rng))
    mu = mu_W(sample, sampler.dimension, sampler.basis)
    if verbose:
        print(f"Initial mu: {mu:.2f}  |  Sample size: {len(sample)}")
    while mu > mu_bound:
        sample.extend(sampler.draw(rng))
        mu = mu_W(sample, sampler.dimension, sampler.basis)
        if verbose:
            print(f"New mu: {mu:.2f}  |  Sample size: {len(sample)}")
    if verbose:
        print("Start subsampling")
    while True:
        candidate = greedy_subsample(sample, len(sample)-1, sampler)
        c_mu = mu_W(candidate, sampler.dimension, sampler.basis)
        if c_mu > mu_bound:
            break
        sample = candidate
        mu = c_mu
        if verbose:
            print(f"New mu: {mu:.2f}  |  Sample size: {len(sample)}")
    return sample, mu

if __name__ == "__main__":
    from pathlib import Path
    from plotting import plt

    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)

    dimension = 10
    rank = 3 * dimension
    discretisation = 1000

    # sample_size = int(2.5 * dimension)
    sample_size = dimension

    rng = np.random.default_rng(0)
    sampler = Sampler(dimension, rank, discretisation)

    sample = []
    while len(sample) < sample_size:
        sample.extend(sampler.draw(rng))
    sample = sample[:sample_size]

    tqdm.write("Start optimisation")
    tqdm.write(f"Initial quasi-optimality factor: {mu_W(sample, sampler.dimension, sampler.basis):.2f}")
    for step in trange(10):
        # sample.extend(sampler.draw(rng))
        # sample = greedy_subsample(sample, sample_size, sampler)
        sample = greedy_swap(sample, sampler.draw(rng))
        tqdm.write(f"[{step}] Quasi-optimality factor: {mu_W(sample, sampler.dimension, sampler.basis):.2f}")

    plot_path = plot_directory / "optimisation_statistics.png"
    if not plot_path.exists():
        trials = 1_000
        samples = []
        mus = []
        for trial in trange(trials):
            sample, mu = greedy_optimise(sample_size, sampler, rng)
            samples.append(sample)
            mus.append(mu)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
        ax[0].hist(np.concatenate(samples), density=True, bins=80)
        ax[0].set_title("Sample distribution")
        ax[1].hist(mus, bins=25)
        ax[1].set_title("Quasi-optimality factor")
        print("Saving optimisation statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    trials = 1_000
    samples = []
    mus = []
    for trial in trange(trials):
        sample, mu = greedy_optimise_mu(2, sampler, rng)
        samples.append(sample)
        mus.append(mu)

    plot_path = plot_directory / "optimisation_statistics_sample_size.png"
    if not plot_path.exists():
        fig, ax = plt.subplots(1, 3, figsize=(8, 4), dpi=300)
        ax[0].hist(np.concatenate(samples), density=True, bins=80)
        ax[0].set_title("Sample distribution")
        ax[1].hist(mus, bins=25)
        ax[1].set_title("Quasi-optimality factor")
        ax[2].hist([len(sample) for sample in samples], bins=25)
        ax[2].set_title("Sample size")
        print("Saving optimisation statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    from legendre import hk_gramian, orthonormal_basis
    from least_squares import optimal_least_squares

    # aspired_target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)
    # target_name = "wave"
    # target_dimension = 14
    # aspired_target = np.exp
    # target_name = "exp"
    # target_dimension = 14
    # target_name = "random"
    # target_dimension = 30
    target_name = "ones"
    target_dimension = 30
    assert target_dimension >= dimension

    xs = np.linspace(-1, 1, 10_000)
    def h1_inner(f, g, xs):
        fx = f(xs)
        gx = g(xs)
        l2_inner = np.trapz(fx * gx, xs)
        dx = np.diff(xs)
        dfx = np.diff(fx) / dx
        dgx = np.diff(gx) / dx
        h10_inner = (dfx * dgx) @ dx
        return l2_inner + h10_inner

    h1_gramian = hk_gramian(target_dimension, 1)
    h1_legendre = orthonormal_basis(h1_gramian)
    if target_name == "random":
        target_coefficients = rng.normal(size=target_dimension)
    elif target_name == "ones":
        target_coefficients = np.ones(target_dimension)
    else:
        target_coefficients = np.zeros(target_dimension)
        e = lambda k: np.eye(1, target_dimension, k=k)[0]
        for dim in range(target_dimension):
            basis_function = lambda x: h1_legendre(x, e(dim))
            target_coefficients[dim] = h1_inner(aspired_target, basis_function, xs)

    target = lambda x: h1_legendre(x, target_coefficients)

    def h1_error(f, g, xs):
        fx = f(xs)
        gx = g(xs)
        l2_error_sqr = np.trapz((fx - gx)**2, xs)
        dx = np.diff(xs)
        dfx = np.diff(fx) / dx
        dgx = np.diff(gx) / dx
        h10_error_sqr = (dfx - dgx)**2 @ dx
        return np.sqrt(l2_error_sqr + h10_error_sqr)

    min_error = np.linalg.norm(target_coefficients[dimension:])
    mus_empirical = []
    for sample in samples:
        coefs = optimal_least_squares(target, sample, sampler.dimension, sampler.basis)
        error = np.sqrt(np.linalg.norm(target_coefficients[:dimension] - coefs)**2 + min_error**2)
        # error <= (1 + 2 * mu) * approximation_error
        # --> (error / approximation_error - 1) / 2 <= mu
        mu_emp = (error / min_error - 1) / 2
        mus_empirical.append(mu_emp)

    plot_path = plot_directory / f"optimal_sampled_least_squares_{target_name}.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    ax.hist(mus_empirical, density=True, bins=25)
    ax.set_title("Empirical quasi-optimality factor")
    print("Saving optimisation statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
