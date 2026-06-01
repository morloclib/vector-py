import numpy as np


def morloc_zeros1(d1):
    return np.zeros((d1,))

def morloc_ones1(d1):
    return np.ones((d1,))

def morloc_fill1(v, d1):
    return np.full((d1,), v)


# Vector <-> List bridge. pack: Vector (numpy.ndarray) -> List (Python
# list). unpack: List -> Vector. Element dtype is decided by numpy's
# default rules from the input list contents.
def morloc_packVector(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def morloc_unpackVector(x):
    if isinstance(x, np.ndarray):
        return x
    xs = list(x)
    if xs and isinstance(xs[0], (bool, int, float, np.bool_, np.integer, np.floating)):
        return np.asarray(xs)
    arr = np.empty(len(xs), dtype=object)
    for i, v in enumerate(xs):
        arr[i] = v
    return arr


# Typeclass-method backends for Vector. numpy arrays are iterable, so
# the same algorithms used for lists work; we keep the result as a
# numpy array where the operation preserves shape.
def morloc_vec_map(f, xs):
    result = [f(x) for x in xs]
    if result and isinstance(result[0], (bool, int, float, np.bool_, np.integer, np.floating)):
        return np.asarray(result)
    arr = np.empty(len(result), dtype=object)
    for i, v in enumerate(result):
        arr[i] = v
    return arr


def morloc_vec_fold(f, b, xs):
    for x in xs:
        b = f(b, x)
    return b


def morloc_vec_fold1(f, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = f(acc, x)
    return acc


def morloc_vec_safeFold1(f, xs):
    if len(xs) == 0:
        return None
    acc = xs[0]
    for x in xs[1:]:
        acc = f(acc, x)
    return acc


# Structural equality for Vector. np.array_equal handles numeric arrays
# (element-wise) and dtype=object arrays (uses Python `==` per element).
# Returns a single bool rather than the element-wise mask that `xs == ys`
# would produce on numpy arrays, so the result is usable as Eq's Bool.
def morloc_vec_eq(xs, ys):
    if len(xs) != len(ys):
        return False
    return bool(np.array_equal(np.asarray(xs), np.asarray(ys)))


# Lexicographic <= on Vector. Returns True iff xs <= ys element-wise
# in lexicographic order. For Vectors of differing length, the shorter
# prefix-matching one compares less.
def morloc_vec_le(xs, ys):
    xs = list(xs)
    ys = list(ys)
    n = min(len(xs), len(ys))
    for i in range(n):
        if xs[i] < ys[i]:
            return True
        if xs[i] > ys[i]:
            return False
    return len(xs) <= len(ys)
