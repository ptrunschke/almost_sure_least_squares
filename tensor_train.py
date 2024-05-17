from __future__ import annotations
from typing import Optional, Literal, Callable
import numpy as np
from jaxtyping import Float, Bool, Int
from opt_einsum import contract

from basis_1d import Basis
from sampling import Density

FVector = Float[np.ndarray, "dimension"]
BVector = Bool[np.ndarray, "dimension"]
TTCore = Float[np.ndarray, "leftRank dimension rightRank"]
RoundingCondition = Callable[[int, FVector], BVector]


class TensorTrain(object):
    def __init__(self, *, from_components: Optional[list[TTCore]] = None, from_tt: Optional[TensorTrain] = None):
        if not ((from_components is None) ^ (from_tt is None)):
            raise ValueError("exactly one of from_components and from_tt must be specified")
        if from_components is not None:
            if not (isinstance(from_components, list) and all(isinstance(cmp, TTCore) for cmp in from_components)):
                raise TypeError("from_components must be a list of TTCore's")
            if (
                not all(cmpl.shape[2] == cmpr.shape[0] for cmpl, cmpr in zip(from_components[:-1], from_components[1:]))
                and from_components[0].shape[0] == 1
                and from_components[-1].shape[-1] == 1
            ):
                raise ValueError("inconsistent shapes of components")
            self._components = [np.array(component) for component in from_components]
            self.canonicalise("left")
        elif from_tt is not None:
            if not isinstance(from_tt, TensorTrain):
                raise TypeError("from_tt must be a TensorTrain")
            self._components = [component.copy() for component in from_tt._components]
            self._core_position = from_tt._core_position
        else:
            raise NotADirectoryError()

    def assert_validity(self):
        assert len(self._components) > 0
        assert all(cmp.ndim == 3 for cmp in self._components)
        assert self._components[0].shape[0] == 1
        assert self._components[-1].shape[-1] == 1
        assert all(cmpl.shape[2] == cmpr.shape[0] for cmpl, cmpr in zip(self._components[:-1], self._components[1:]))
        assert 0 <= self._core_position < len(self._components)
        for m, cmp in enumerate(self._components[:self._core_position]):
            cmp = cmp.reshape(-1, cmp.shape[-1])
            assert np.linalg.norm(cmp.T @ cmp - np.eye(cmp.shape[-1])) <= 1e-12, f"component {m} (core position {self.core_position})"
        for m, cmp in enumerate(self._components[self._core_position + 1:], start=self._core_position + 1):
            cmp = cmp.reshape(cmp.shape[0], -1)
            assert np.linalg.norm(cmp @ cmp.T - np.eye(cmp.shape[0])) <= 1e-12, f"component {m} (core position {self.core_position})"

    @classmethod
    def random(cls, rng: np.random.Generator, dimensions: list[int], ranks: list[int]) -> TensorTrain:
        assert 0 < len(dimensions) == len(ranks) - 1
        assert all(d > 0 for d in dimensions)
        assert all(r > 0 for r in ranks) and ranks[0] == ranks[-1] == 1
        components = [rng.standard_normal(size=(rl, d, rr)) for rl, d, rr in zip(ranks, dimensions, ranks[1:])]
        return cls(from_components=components)

    @property
    def order(self) -> int:
        return len(self._components)

    @property
    def dimensions(self) -> list[int]:
        return [component.shape[1] for component in self._components]

    @property
    def ranks(self) -> list[int]:
        return [1] + [component.shape[2] for component in self._components]

    @property
    def parameters(self) -> int:
        return sum(component.size for component in self._components)

    def move_core(
        self, direction: Literal["left"] | Literal["right"], rounding_condition: Optional[RoundingCondition] = None
    ) -> tuple[FVector, BVector]:
        """Move the core in the specified direction.

        Parameters
        ----------
        direction : "left" | "right"
            The direction to move into.
        rounding_condition : Callable, optional
            A function taking an edge position (indexed as in the ranks() tuple) and a vector of singular values.
            The return value of the function should be a boolean array of the same length as the singular values.
            Only singular values for which the corresponding entry in the returned array is True are kept.
            If no rounding condition is specified, all singular values are kept.

        Returns
        -------
        FVector
            The singular values that were kept.
        BVector
            The mask of kept singular values.

        Raises
        ------
        TypeError
            If the core is already at the edge in the specified direction.
        """
        if rounding_condition is None:
            rounding_condition = lambda p, s: np.full(len(s), True)
        if direction == "left":
            if self._core_position == 0:
                raise ValueError('move_core("left") impossible at core position 0')
            core = self._components[self._core_position]
            l, e, r = core.shape
            core = core.reshape(l, e * r)
            u, s, vt = np.linalg.svd(core, full_matrices=False)
            # assert np.allclose((u * s) @ vt, core)
            mask = rounding_condition(self._core_position, s)
            if not np.any(mask):
                raise RuntimeError("rounding produces rank zero")
            u, s, vt = u[:, mask], s[mask], vt[mask]
            self._components[self._core_position] = vt.reshape(len(s), e, r)
            self._core_position -= 1
            self._components[self._core_position] = contract(
                "leR, Rr -> ler", self._components[self._core_position], u * s
            )
        elif direction == "right":
            if self._core_position == self.order - 1:
                raise ValueError(f'move_core("right") impossible at core position {self._core_position}')
            core = self._components[self._core_position]
            l, e, r = core.shape
            core = core.reshape(l * e, r)
            u, s, vt = np.linalg.svd(core, full_matrices=False)
            # assert np.allclose(u @ (s[:,None] * vt), core)
            mask = rounding_condition(self._core_position + 1, s)
            if not np.any(mask):
                raise RuntimeError("rounding produces rank zero")
            u, s, vt = u[:, mask], s[mask], vt[mask]
            self._components[self._core_position] = u.reshape(l, e, len(s))
            self._core_position += 1
            self._components[self._core_position] = contract(
                "lL, Ler -> ler", s[:, None] * vt, self._components[self._core_position]
            )
        else:
            raise TypeError(f'unknown direction. Expected "left" or "right" but got "{direction}"')
        return s, mask

    def canonicalise(
        self, side: Literal["left"] | Literal["right"], rounding_condition: Optional[RoundingCondition] = None
    ):
        self._core_position = {"left": self.order - 1, "right": 0}[side]
        limit = {"left": 0, "right": self.order - 1}[side]
        while self._core_position != limit:
            self.move_core(side, rounding_condition)

    def round(self, rounding_condition: RoundingCondition):
        """
        Round the tensor train according to the specified condition.

        Parameters
        ----------
        rounding_condition : callable
            Should take a numpy.ndarray of singular values and return a boolean numpy.ndarray specifying if the
            corresponding singular value should be kept.
        """
        self.canonicalise("left")
        self.canonicalise("right", rounding_condition)

    def singular_values(self) -> list[FVector]:
        copy = TensorTrain(from_tt=self)
        copy.canonicalise("left")
        singularValues = []
        while copy._core_position != copy.order - 1:
            singularValues.append(copy.move_core("right")[0])
        norm = np.array([np.linalg.norm(singularValues[0])])
        return [norm] + singularValues + [norm]

    @property
    def core(self) -> TTCore:
        return self._components[self._core_position]

    @core.setter
    def core(self, value: TTCore):
        if not isinstance(value, TTCore):
            raise TypeError("value must be a TTCore")
        if not value.shape == self._components[self._core_position].shape:
            raise ValueError(
                f"inconsistent shape. Expected {self._components[self._core_position].shape} but got {value.shape}"
            )
        self._components[self._core_position] = value

    @property
    def core_position(self) -> int:
        return self._core_position

    def sample_from_square(self, rng: np.random.Generator, sample_size: int = 1) -> Int[np.ndarray, " dimension"]:
        """Sample from the square of the tensor train.

        Notes
        -----
        Sampling from f(x, y, z)^2 given x and conditioning on z:
          p(x, y) = \int f(x, y, z)^2 dz
          p(y | x) = p(x, y) / p(x) ∝ \int f(x, y, z)^2 dz
        Suppose that f(x, y, z) = C1(x) C2(y) C3(z) then
                  p(y | x) ∝ C1(x) C2(y) (\int C3(z) C3(z).T dz) C2(y).T C1(x).T
                           = C1(x) C2(y) C2(y).T C1(x).T .
        """
        sqrt_density = TensorTrain(from_tt=self)
        while sqrt_density.core_position != 0:
            sqrt_density.move_core("left")

        conditioning = np.ones((sample_size, 1))
        sample = np.empty((sample_size, self.order), dtype=int)
        for position in range(self.order):
            # Condition on the preceding variables and marginalise over the subsequent variables.
            component = sqrt_density._components[position]
            pdf = contract("nl, ler -> ner", conditioning, component)
            pdf = contract("ner, ner -> ne", pdf, pdf)
            cdf = np.cumsum(pdf, axis=1)
            cdf /= cdf[:, -1][:, None]
            us = rng.random(size=(sample_size,))
            sample[:, position] = np.count_nonzero(cdf < us[:, None], axis=1)
            conditioning = contract("nl, lnr -> nr", conditioning, component[:, sample[:, position], :])
        return sample

    def transform(self, rank_one_transform: list[Float[np.ndarray, "new_dimension old_dimension"]]) -> TensorTrain:
        return TensorTrainCoreSpace(self).transform(rank_one_transform)[0].tensor_train

    def evaluate(self, rank_one_measurement: list[Float[np.ndarray, "dimension sample_size"]]) -> Float[np.ndarray, " sample_size"]:
        ln, en, rn = TensorTrainCoreSpace(self).evaluate(rank_one_measurement)
        return contract("ler, ln, en, rn -> n", self.core, ln, en, rn)

    def split(self) -> tuple[TensorTrain, TTCore, TensorTrain]:
        left_tt = TensorTrain(from_tt=self)
        right_tt = TensorTrain(from_tt=self)
        core = left_tt.core

        del left_tt._components[left_tt._core_position + 1:]
        left_tt._components[left_tt._core_position] = np.eye(left_tt.core.shape[0])[:, :, None]
        assert left_tt.order == self.core_position + 1
        assert left_tt.core_position == self.core_position
        assert np.all(left_tt.core == np.eye(self.core.shape[0])[:, :, None])
        assert all(np.all(left_tt._components[pos] == self._components[pos]) for pos in range(self.core_position))

        del right_tt._components[: right_tt._core_position]
        right_tt._core_position = 0
        right_tt._components[right_tt._core_position] = np.eye(right_tt.core.shape[2])[None, :, :]
        assert right_tt.order == self.order - self.core_position
        assert right_tt.core_position == 0
        assert np.all(right_tt.core == np.eye(self.core.shape[2])[None, :, :])
        assert all(np.all(right_tt._components[pos] == self._components[self.core_position + pos]) for pos in range(1, self.order - self.core_position))

        return left_tt, core, right_tt

    def transpose(self) -> TensorTrain:
        result = TensorTrain(from_tt=self)
        result._components = [np.transpose(component) for component in reversed(result._components)]
        result._core_position = result.order - result._core_position - 1
        return result

    def fix_core(self, *, at: int, direction: Optional[Literal["left"] | Literal["right"]] = None) -> TensorTrain:
        "Contract a standard basis vector to the core of the tensor train."
        assert self.order > 1
        assert 0 <= at < self.core.shape[1]
        if direction is None:
            if self.core_position < self.order - 1:
                direction = "right"
            else:
                direction = "left"
        assert direction in ["left", "right"]
        new_core_position = self.core_position + {"left": -1, "right": 1}[direction]
        assert 0 <= new_core_position < self.order
        result = TensorTrain(from_tt=self)
        if direction == "right":
            new_core = result._components[new_core_position]
            new_core = contract("lL, Ler -> ler", result.core[:, at, :], new_core)
            result._components[new_core_position] = new_core
            del result._components[result._core_position]
        else:
            new_core = result._components[new_core_position]
            new_core = contract("leR, Rr -> ler", new_core, result.core[:, at, :])
            result._components[new_core_position] = new_core
            del result._components[result._core_position]
            result.core_position -= 1
        result.assert_validity()
        return result


class TensorTrainCoreSpace(object):
    def __init__(self, tensor_train: TensorTrain):
        self.tensor_train = tensor_train

    def evaluate(self, rank_one_measurement: list[Float[np.ndarray, "dimension sample_size"]]) -> tuple[
        Float[np.ndarray, "left_rank sample_size"],
        Float[np.ndarray, "dimension sample_size"],
        Float[np.ndarray, "right_rank sample_size"],
    ]:
        """
        Compute the measurements on the core space from the rank-one measurements on the full tensor space.

        Parameters
        ----------
        univariate_measurements : list of numpy.ndarray
            The component tensors for the rank-one measurement operator on the full tensor space.
        """
        if not len(rank_one_measurement) == self.tensor_train.order:
            raise ValueError("inconsistent number of univariate measurements")
        sample_size = rank_one_measurement[0].shape[1]
        if not all(
            measure.shape == (dimension, sample_size)
            for measure, dimension in zip(rank_one_measurement, self.tensor_train.dimensions)
        ):
            raise ValueError("inconsistent dimensions of univariate measurements")

        left_space_measurement = np.ones((1, sample_size))
        for measure, component in zip(
            rank_one_measurement, self.tensor_train._components[: self.tensor_train.core_position]
        ):
            left_space_measurement = contract("ln, en, ler -> rn", left_space_measurement, measure, component)
        assert left_space_measurement.shape == (
            self.tensor_train.core.shape[0],
            sample_size,
        )

        right_space_measurement = np.ones((1, sample_size))
        for measure, component in zip(
            reversed(rank_one_measurement),
            reversed(self.tensor_train._components[self.tensor_train._core_position + 1 :]),
        ):
            right_space_measurement = contract("ler, en, rn -> ln", component, measure, right_space_measurement)
        assert right_space_measurement.shape == (
            self.tensor_train.core.shape[2],
            sample_size,
        )

        return left_space_measurement, rank_one_measurement[self.tensor_train._core_position], right_space_measurement

    def transform(self, rank_one_transform: list[Float[np.ndarray, "new_dimension old_dimension"]]) -> tuple[
        TensorTrainCoreSpace,
        tuple[
            Float[np.ndarray, "new_left_rank old_left_rank"],
            Float[np.ndarray, "new_dimension old_dimension"],
            Float[np.ndarray, "new_right_rank old_right_rank"],
        ],
    ]:
        """
        Returns
        -------
        TensorTrainCoreSpace
            The transformed core space.
        np.ndarray
            The transformation that has to be performed on the coeffients.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> space = TensorTrainCoreSpace(TensorTrain.random(rng, dimensions=[10, 10], ranks=[5]))
        >>> transform = rng.standard_normal(size=(2, 10, 10))
        >>> C = space.tensor_train.core
        >>> new_space, (lT, eT, rT) = space.transform(transform)
        >>> new_C = space.tensor_train.core
        >>> assert np.allclose(new_C, contract("LER, lL, eE, rR -> ler", C, lT, eT, rT))
        """
        if not len(rank_one_transform) == self.tensor_train.order:
            raise ValueError("inconsistent number of univariate measurements")
        inconsistent_factors = []
        for position in range(self.tensor_train.order):
            transform = rank_one_transform[position]
            dimension = self.tensor_train.dimensions[position]
            if transform.ndim != 2 or transform.shape[1] != dimension:
                inconsistent_factors.append(str(position))
        if len(inconsistent_factors) > 0:
            raise ValueError(f"inconsistent dimensions of univariate measurement(s) {', '.join(inconsistent_factors)}")

        result = TensorTrain(from_tt=self.tensor_train)
        for position in range(result.order):
            result._components[position] = contract(
                "de, ler -> ldr", rank_one_transform[position], result._components[position]
            )

        core_position = result._core_position
        core = result.core

        result._components[core_position] = np.eye(core.shape[0])[:, :, None]
        result._core_position = 0
        while result.core_position < core_position:
            result.move_core("right")
        assert result.core_position == core_position
        left_transform = result._components[core_position][:, :, 0]

        result._components[core_position] = np.eye(core.shape[2])[None, :, :]
        result._core_position = result.order - 1
        while result.core_position > core_position:
            result.move_core("left")
        assert result.core_position == core_position
        right_transform = result._components[core_position][0, :, :]

        core = contract("lEr, Ll, Rr -> LER", core, left_transform, right_transform)
        result._components[core_position] = core
        result.assert_validity()
        return TensorTrainCoreSpace(result), (left_transform, rank_one_transform[core_position], right_transform)

    def christoffel(self, points: Float[np.ndarray, "sample_size dimension"], bases: list[Basis]) -> Float[np.ndarray, " sample_size"]:
        points = np.asarray(points)
        assert points.ndim == 2 and points.shape[1] == len(bases)
        assert len(bases) == self.tensor_train.order
        assert all(basis.dimension == dimension for basis, dimension in zip(bases, self.tensor_train.dimensions))
        bs = [basis(points[:, pos]) for pos, basis in enumerate(bases)]
        ln, en, rn = self.evaluate(bs)
        core = self.tensor_train.core
        assert ln.shape == (core.shape[0], points.shape[0]) and en.shape == (core.shape[1], points.shape[0]) and rn.shape == (core.shape[2], points.shape[0])
        return np.sum(ln**2, axis=0) * np.sum(en**2, axis=0) * np.sum(rn**2, axis=0) / core.size

    def christoffel_sample(
        self, rng: np.random.Generator, bases: list[Basis], densities: list[Density], discretisations: list[FVector], sample_size: int, stratified: bool = False
    ) -> Float[np.ndarray, "sample_size dimension"]:
        assert len(bases) == len(densities) == len(discretisations) == self.tensor_train.order
        assert all(basis.dimension == dimension for basis, dimension in zip(bases, self.tensor_train.dimensions))
        assert sample_size > 0
        assert all(discretisation.ndim == 1 for discretisation in discretisations)

        core_position = self.tensor_train.core_position
        transforms = lambda pos: (bases[pos](discretisations[pos]) * np.sqrt(densities[pos](discretisations[pos]))).T
        transforms = [transforms(pos) for pos in range(self.tensor_train.order)]
        left_sqrt_density, _, right_sqrt_density = self.tensor_train.split()
        left_sqrt_density = left_sqrt_density.transform(transforms[:core_position] + [np.eye(left_sqrt_density.core.shape[1])])
        left_sqrt_density = left_sqrt_density.transpose()
        right_sqrt_density = right_sqrt_density.transform([np.eye(right_sqrt_density.core.shape[1])] + transforms[core_position + 1:])
        assert left_sqrt_density.core_position == right_sqrt_density.core_position == 0

        left_sample = np.empty((sample_size, left_sqrt_density.order - 1), dtype=int)
        if left_sample.size > 0:
            left_rank = left_sqrt_density.dimensions[0]
            if stratified:
                left_sample_sizes = np.full(left_rank, sample_size // left_rank, dtype=int)
                left_sample_sizes[rng.choice(left_rank, size=sample_size % left_rank, replace=False)] += 1
            else:
                left_sample_sizes = np.zeros(left_rank, dtype=int)
            for index in rng.choice(left_rank, size=sample_size):
                left_sample_sizes[index] += 1
            assert np.sum(left_sample_sizes) == sample_size
            start = 0
            for index, sample_size_index in enumerate(left_sample_sizes):
                left_sample_index = left_sqrt_density.fix_core(at=index).sample_from_square(rng, sample_size=sample_size_index)
                stop = start + sample_size_index
                left_sample[start : stop] = left_sample_index
                start = stop
            assert start == sample_size
            left_sample = left_sample[:, ::-1]
        rng.shuffle(left_sample, axis=0)

        right_sample = np.empty((sample_size, right_sqrt_density.order - 1), dtype=int)
        if right_sample.size > 0:
            right_rank = right_sqrt_density.dimensions[0]
            if stratified:
                right_sample_sizes = np.full(right_rank, sample_size // right_rank, dtype=int)
                right_sample_sizes[rng.choice(right_rank, size=sample_size % right_rank, replace=False)] += 1
            else:
                right_sample_sizes = np.zeros(right_rank, dtype=int)
            for index in rng.choice(right_rank, size=sample_size):
                right_sample_sizes[index] += 1
            assert np.sum(right_sample_sizes) == sample_size
            start = 0
            for index, sample_size_index in enumerate(right_sample_sizes):
                right_sample_index = right_sqrt_density.fix_core(at=index).sample_from_square(rng, sample_size=sample_size_index)
                stop = start + sample_size_index
                right_sample[start : stop] = right_sample_index
                start = stop
            assert start == sample_size
        rng.shuffle(right_sample, axis=0)

        christoffel = np.sum(bases[core_position](discretisations[core_position])**2, axis=0) * densities[core_position](discretisations[core_position])
        christoffel = christoffel / np.sum(christoffel)
        middle_sample = rng.choice(len(discretisations[core_position]), size=sample_size, p=christoffel)
        assert middle_sample.shape == (sample_size,)

        sample = np.concatenate([left_sample, middle_sample[:, None], right_sample], axis=1)
        assert sample.shape == (sample_size, self.tensor_train.order)
        sample = np.array([d[i] for d, i in zip(discretisations, sample.T)]).T
        assert sample.shape == (sample_size, self.tensor_train.order)
        return sample
