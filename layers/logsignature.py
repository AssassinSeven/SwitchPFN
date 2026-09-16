"""Order-2/3 logsignatures in a standard Lyndon basis (NumPy only)."""

import functools

import numpy as np

LIE_PROJECTION_ABSOLUTE_TOLERANCE = 2e-11
LIE_PROJECTION_RELATIVE_TOLERANCE = 2e-13


class LogSignatureViolation(RuntimeError):
    pass


def witt_dimension(dimension: int, order: int) -> int:
    """Cumulative free-Lie dimension through ``order`` (Witt formula)."""

    if (
        type(dimension) is not int
        or dimension < 1
        or type(order) is not int
        or order not in (2, 3)
    ):
        raise LogSignatureViolation(
            "dimension/order outside supported logsignature domain"
        )
    degree_two = (dimension * dimension - dimension) // 2
    if order == 2:
        return dimension + degree_two
    degree_three = (dimension**3 - dimension) // 3
    return dimension + degree_two + degree_three


def _is_lyndon(word: tuple[int, ...]) -> bool:
    # A word is strictly smaller than each nontrivial rotation.
    return all(word < word[index:] + word[:index] for index in range(1, len(word)))


@functools.lru_cache(maxsize=None)
def lyndon_words(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    if (
        type(dimension) is not int
        or dimension < 1
        or type(degree) is not int
        or degree < 1
    ):
        raise LogSignatureViolation("invalid Lyndon alphabet/degree")
    import itertools

    return tuple(
        word
        for word in itertools.product(range(dimension), repeat=degree)
        if _is_lyndon(word)
    )


def _add(
    left: dict[tuple[int, ...], float],
    right: dict[tuple[int, ...], float],
    scale: float = 1.0,
):
    output = dict(left)
    for word, value in right.items():
        output[word] = output.get(word, 0.0) + scale * value
    return {word: value for word, value in output.items() if value != 0.0}


def _product(left: dict[tuple[int, ...], float], right: dict[tuple[int, ...], float]):
    output: dict[tuple[int, ...], float] = {}
    for a, av in left.items():
        for b, bv in right.items():
            output[a + b] = output.get(a + b, 0.0) + av * bv
    return output


@functools.lru_cache(maxsize=None)
def standard_lyndon_bracket(word: tuple[int, ...]) -> dict[tuple[int, ...], float]:
    """Tensor expansion of the standard Lyndon bracketing ``[word]``."""

    if not _is_lyndon(word):
        raise LogSignatureViolation("standard bracketing requires a Lyndon word")
    if len(word) == 1:
        return {word: 1.0}
    # Standard factorization: v is the longest proper Lyndon suffix, w = uv.
    suffix = None
    for index in range(1, len(word)):
        candidate = word[index:]
        if _is_lyndon(candidate):
            suffix = candidate
            break
    if suffix is None:
        raise LogSignatureViolation("Lyndon standard factorization failed")
    prefix = word[: len(word) - len(suffix)]
    left, right = standard_lyndon_bracket(prefix), standard_lyndon_bracket(suffix)
    return _add(_product(left, right), _product(right, left), -1.0)


@functools.lru_cache(maxsize=None)
def _basis(
    dimension: int, degree: int
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    words = lyndon_words(dimension, degree)
    all_words = tuple(__import__("itertools").product(range(dimension), repeat=degree))
    row = {word: index for index, word in enumerate(all_words)}
    matrix = np.zeros((dimension**degree, len(words)), dtype=np.float64)
    for column, word in enumerate(words):
        for tensor_word, coefficient in standard_lyndon_bracket(word).items():
            matrix[row[tensor_word], column] = coefficient
    if np.linalg.matrix_rank(matrix) != len(words):
        raise LogSignatureViolation("Lyndon bracket basis is rank deficient")
    return words, matrix


@functools.lru_cache(maxsize=None)
def _basis_left_inverse(dimension: int, degree: int) -> np.ndarray:
    """Cached left inverse for repeated projection of tensor-log levels."""

    _words, matrix = _basis(dimension, degree)
    inverse = np.linalg.pinv(matrix)
    identity = inverse @ matrix
    if not np.allclose(identity, np.eye(matrix.shape[1]), atol=2e-13, rtol=2e-13):
        raise LogSignatureViolation("Lyndon basis left inverse failed audit")
    inverse.setflags(write=False)
    return inverse


def tensor_signature_levels(
    increments: np.ndarray, order: int
) -> tuple[np.ndarray, ...]:
    values = _increments(increments, order)
    dimension = values.shape[1]
    count = len(values)
    if count == 0:
        empty = [np.zeros(dimension), np.zeros((dimension, dimension))]
        if order == 3:
            empty.append(np.zeros((dimension, dimension, dimension)))
        return tuple(empty)

    # Vectorized Chen recurrence.  prefix_one[n] and prefix_two[n] are the
    # signature levels immediately *before* multiplying by exp(dx_n).
    cumulative_one = np.cumsum(values, axis=0)
    prefix_one = np.concatenate((np.zeros((1, dimension)), cumulative_one[:-1]), axis=0)
    same_step_two = np.einsum("ni,nj->nij", values, values) / 2.0
    delta_two = np.einsum("ni,nj->nij", prefix_one, values) + same_step_two
    second = np.sum(delta_two, axis=0)
    first = _stable_endpoint(values)
    if order == 2:
        return first, second

    cumulative_two = np.cumsum(delta_two, axis=0)
    prefix_two = np.concatenate(
        (np.zeros((1, dimension, dimension)), cumulative_two[:-1]),
        axis=0,
    )
    third = np.sum(
        np.einsum("nij,nk->nijk", prefix_two, values)
        + np.einsum("ni,njk->nijk", prefix_one, same_step_two)
        + np.einsum("ni,nj,nk->nijk", values, values, values) / 6.0,
        axis=0,
    )
    return first, second, third


def tensor_log_levels(increments: np.ndarray, order: int) -> tuple[np.ndarray, ...]:
    signature = tensor_signature_levels(increments, order)
    first, second = signature[0], signature[1]
    output = [first, second - np.einsum("i,j->ij", first, first) / 2.0]
    if order == 3:
        third = signature[2]
        third = (
            third
            - (
                np.einsum("i,jk->ijk", first, second)
                + np.einsum("ij,k->ijk", second, first)
            )
            / 2.0
            + np.einsum("i,j,k->ijk", first, first, first) / 3.0
        )
        output.append(third)
    return tuple(output)


def _increments(value: np.ndarray, order: int) -> np.ndarray:
    if type(order) is not int or order not in (2, 3):
        raise LogSignatureViolation("only orders 2 and 3 are supported")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise LogSignatureViolation("increments must be a finite 2D matrix")
    return np.ascontiguousarray(array)


def _stable_endpoint(increments: np.ndarray) -> np.ndarray:
    """Sum endpoint increments in a stable, deterministic order."""
    values = np.asarray(increments, dtype=np.float64)
    return np.asarray(
        [
            sum(sorted(values[:, index].tolist(), key=lambda item: (abs(item), item)))
            for index in range(values.shape[1])
        ],
        dtype=np.float64,
    )


def logsignature(increments: np.ndarray, order: int) -> np.ndarray:
    levels = tensor_log_levels(increments, order)
    dimension = levels[0].size
    coordinates = []
    for degree, tensor in enumerate(levels, start=1):
        _words, matrix = _basis(dimension, degree)
        solution = _basis_left_inverse(dimension, degree) @ tensor.reshape(-1)
        flattened = tensor.reshape(-1)
        residual = matrix @ solution - flattened
        # A scale-aware residual bound tolerates cancellation on large paths
        # while checking that the tensor log lies in the free Lie algebra.
        scale = max(1.0, float(np.max(np.abs(flattened))))
        maximum_residual = float(np.max(np.abs(residual)))
        allowed_residual = (
            LIE_PROJECTION_ABSOLUTE_TOLERANCE
            + LIE_PROJECTION_RELATIVE_TOLERANCE * scale
        )
        if not np.isfinite(maximum_residual) or maximum_residual > allowed_residual:
            raise LogSignatureViolation(
                "tensor logarithm is not in the truncated free Lie algebra"
            )
        coordinates.append(solution)
    output = np.concatenate(coordinates)
    if output.size != witt_dimension(dimension, order) or not np.isfinite(output).all():
        raise LogSignatureViolation(
            "compressed log-signature width/finite guard failed"
        )
    return output


def multiscale_logsignature(increments, order=3, partitions=(1, 2, 4)):
    """Concatenate logsignatures on consecutive partitions of the increments."""
    return np.concatenate(
        [
            logsignature(increments[index], order)
            for count in partitions
            for index in np.array_split(np.arange(len(increments)), count)
        ]
    )
