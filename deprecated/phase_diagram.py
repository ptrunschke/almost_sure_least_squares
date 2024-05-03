if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm, trange

    from rkhs import rkhs_kernel
    from sampling_new import Sampler, draw_dpp_sample, draw_subspace_dpp_sample
    from legendre import hk_gramian, orthonormal_basis
    from least_squares import quasi_optimality_constant, bias_constant

    plot_directory = Path(__file__).parent / "plot" / "phase_diagram"
    plot_directory.mkdir(exist_ok=True)

    dimensions = np.linspace(5, 50, num=10, dtype=int)
    sample_sizes = np.linspace(dimensions[0], 2 * dimensions[-1], num=11, dtype=int)
    # sample_sizes = np.linspace(10, 510, num=11, dtype=int)
    trials = 10
    # dimensions = np.linspace(5, 15, num=3, dtype=int)
    # sample_sizes = np.linspace(10, 110, num=3, dtype=int)
    # trials = 3
    discretisation = np.linspace(-1, 1, 1_000)

    print("Dimensions:", dimensions)
    print("Sample sizes:", sample_sizes)
    print("Trials:", trials)

    def l2_gramian_cond(points, dimension, *, basis=None):
        if basis is None:
            basis = orthonormal_basis(hk_gramian(dimension, 0))
        M = basis(points, np.eye(dimension)) / np.sqrt(len(points))
        return np.linalg.cond(M)

    def compute_constants(sampling_class, dimensions, sample_sizes, trials, cache_path):
        values = np.full((3, len(dimensions), len(sample_sizes), trials), np.nan)
        if cache_path.exists():
            z = np.load(cache_path)
            existing_dimensions = z["dimensions"]
            existing_sample_sizes = z["sample_sizes"]
            existing_values = z["values"]
            common_trials = min(trials, existing_values.shape[-1])
            for idx_dim, dimension in enumerate(dimensions):
                if dimension not in existing_dimensions:
                    continue
                existing_idx_dim = np.where(existing_dimensions == dimension)[0][0]
                for idx_spl, sample_size in enumerate(sample_sizes):
                    if sample_size not in existing_sample_sizes:
                        continue
                    existing_idx_spl = np.where(existing_sample_sizes == sample_size)[0][0]
                    values[:, idx_dim, idx_spl, :common_trials] = existing_values[:, existing_idx_dim, existing_idx_spl, :common_trials]
                # TODO: Don't overwrite values! Create a larger values arrar here!

        for idx_dim, dimension in enumerate(dimensions):
            tqdm.write(f"Dimension: {dimension}")
            sampler = sampling_class(dimension)
            subspace_basis = orthonormal_basis(hk_gramian(dimension, 1))
            l2_subspace_basis = orthonormal_basis(hk_gramian(dimension, 0))
            for idx_spl, sample_size in enumerate(sample_sizes):
                tqdm.write(f"  Sample size: {sample_size}")
                if sample_size < dimension:
                    values[:, idx_dim, idx_spl] = np.inf
                    continue
                rng = np.random.default_rng(0)
                for trial in trange(trials):
                    if not np.any(np.isnan(values[:, idx_dim, idx_spl, trial])):
                        continue
                    sample = sampler.draw(rng, sample_size)
                    values[0, idx_dim, idx_spl, trial] = quasi_optimality_constant(sample, dimension, basis=subspace_basis)
                    values[1, idx_dim, idx_spl, trial] = bias_constant(sample, dimension, basis=subspace_basis)
                    values[2, idx_dim, idx_spl, trial] = l2_gramian_cond(sample, dimension, basis=l2_subspace_basis)
                np.savez_compressed(cache_path, values=values, dimensions=dimensions, sample_sizes=sample_sizes)

        return values

    class FullKernelSampling:
        def __init__(self, dimension):
            self.sampler = Sampler(dimension, discretisation)

        def draw(self, rng, sample_size):
            return self.sampler.draw(rng, sample_size)

    class RepeatedKernelSampling:
        def __init__(self, dimension):
            self.dimension = dimension
            self.sampler = Sampler(dimension, discretisation)

        def draw(self, rng, sample_size):
            sample = []
            while len(sample) < sample_size:
                sample = np.concatenate([sample, self.sampler.draw(rng, self.dimension)])
            return sample[:sample_size]

    class RepeatedDPPSampling:
        def __init__(self, dimension):
            self.dimension = dimension

        def draw(self, rng, sample_size):
            sample = []
            while len(sample) < sample_size:
                partial_sample = []
                for _ in range(self.dimension):
                    partial_sample.append(draw_dpp_sample(rng, rkhs_kernel, self.dimension, discretisation, conditioned_on=partial_sample))
                sample = np.concatenate([sample, partial_sample])
            return sample[:sample_size]

    class RepeatedSubspaceDPPSampling:
        def __init__(self, dimension):
            self.dimension = dimension
            self.subspace_basis = orthonormal_basis(hk_gramian(dimension, 0))
            # self.subspace_basis = orthonormal_basis(hk_gramian(dimension, 1))

        def draw(self, rng, sample_size):
            sample = []
            while len(sample) < sample_size:
                partial_sample = []
                for _ in range(self.dimension):
                    partial_sample.append(draw_subspace_dpp_sample(rng, self.subspace_basis, self.dimension, discretisation, conditioned_on=partial_sample))
                sample = np.concatenate([sample, partial_sample])
            return sample[:sample_size]

    import re

    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    sampling_class = FullKernelSampling
    # sampling_class = RepeatedKernelSampling
    # sampling_class = RepeatedSubspaceDPPSampling
    # sampling_class = RepeatedDPPSampling

    for sampling_class in [FullKernelSampling, RepeatedKernelSampling, RepeatedSubspaceDPPSampling, RepeatedDPPSampling]:
        sampling_name = camel_to_snake(sampling_class.__name__).split("_")
        assert sampling_name[-1] == "sampling"
        sampling_name = "_".join(sampling_name[:-1])
        values = compute_constants(sampling_class, dimensions, sample_sizes, trials, plot_directory / f"{sampling_name}.npz")

        # cm = mpl.colormaps["viridis"]
        # cm = mpl.colors.LinearSegmentedColormap.from_list("grassland", cm(np.linspace(0, 0.85, 256)[::-1]))
        # nm = mpl.colors.LogNorm()
        # coq_values = np.median(values[0], axis=-1)
        # bc_values = np.median(values[1], axis=-1)

        cm = mpl.colormaps["viridis"]
        cm = mpl.colors.LinearSegmentedColormap.from_list("grassland", cm(np.linspace(0, 0.85, 256)))
        nm = None
        coq_values = np.mean(values[0] <= 6, axis=-1)
        bc_values = np.mean(values[1] <= 75, axis=-1)
        gc_values = np.mean(values[2] <= 10, axis=-1)

        fig, ax = plt.subplots(1, 3)
        Ds, Ss = np.meshgrid(dimensions, sample_sizes, indexing="xy")

        sc = ax[0].scatter(Ds, Ss, c=coq_values.T, s=12, edgecolors='k', linewidths=0.5, norm=nm, cmap=cm, zorder=2)
        # cax = make_axes_locatable(ax[0]).append_axes('right', size='5%', pad=0.05)
        # cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
        # cbar.set_label("Median error", rotation=270, labelpad=15)

        sc = ax[1].scatter(Ds, Ss, c=bc_values.T, s=12, edgecolors='k', linewidths=0.5, norm=nm, cmap=cm, zorder=2)
        # cax = make_axes_locatable(ax[1]).append_axes('right', size='5%', pad=0.05)
        # cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
        # cbar.set_label("Median error", rotation=270, labelpad=15)

        sc = ax[2].scatter(Ds, Ss, c=gc_values.T, s=12, edgecolors='k', linewidths=0.5, norm=nm, cmap=cm, zorder=2)
        # cax = make_axes_locatable(ax[2]).append_axes('right', size='5%', pad=0.05)
        # cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
        # cbar.set_label("Median error", rotation=270, labelpad=15)

        fig.suptitle(sampling_name.replace("_", " ").capitalize() + " sampling")

    plt.show()

    exit()

    sampler = Sampler(dimension, discretisation)
    repeated_samples = []
    for trial in trange(trials):
        sample = []
        while len(sample) < sample_size:
            sample = np.concatenate([sample, sampler.draw(rng, dimension)])
        sample = sample[:sample_size]
        repeated_samples.append(sample)
    plot_path = plot_directory / f"repeated_sample_statistics.png"
