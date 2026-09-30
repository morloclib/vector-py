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


# Element access via Indexable. Index arrives as ?Int64 to match
# __to_index__'s return shape; a None index has no semantic meaning at
# runtime. Python's native list indexing already wraps negative indices
# from the end, but explicit normalization keeps the semantics aligned
# with the C++ and R instances (and with root-py's morloc_at).
def morloc_vec_at(i, xs):
    if i is None:
        raise IndexError("morloc_vec_at: index is Null")
    if i < 0:
        i += len(xs)
    return xs[i]


# Python-style slice with optional bounds. start/stop/step may each be
# None (passed as morloc Null). step 0 is a runtime error per the
# Sliceable contract.
def morloc_vec_slice(start, stop, step, xs):
    if step == 0:
        raise ValueError("slice step cannot be zero")
    return xs[start:stop:step]


def morloc_vec_size(xs):
    return len(xs)


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


# Concatenation for SemigroupDim: the result length is the sum of the two
# input lengths. Python's own `+` cannot serve here -- on a numpy array it
# is elementwise addition, which requires matching lengths and returns one
# of them, contradicting `(++) :: f m a -> f n a -> f (m+n) a`.
def morloc_vec_concat(xs, ys):
    return np.concatenate((np.asarray(xs), np.asarray(ys)))


# Elementwise arithmetic for Integral / Numeric on Vector. numpy applies
# each of these across the whole array in one vectorized call, which is the
# point: the alternative is a round trip through a Python list, which costs
# an object per element in both directions.
def morloc_vec_neg(xs):
    return np.negative(xs)

def morloc_vec_abs(xs):
    return np.absolute(xs)

def morloc_vec_add(xs, ys):
    return np.add(xs, ys)

def morloc_vec_sub(xs, ys):
    return np.subtract(xs, ys)

def morloc_vec_mul(xs, ys):
    return np.multiply(xs, ys)

def morloc_vec_floordiv(xs, ys):
    return np.floor_divide(xs, ys)

def morloc_vec_mod(xs, ys):
    return np.mod(xs, ys)

def morloc_vec_pow(xs, ys):
    return np.power(xs, ys)

def morloc_vec_div(xs, ys):
    return np.divide(xs, ys)

def morloc_vec_inv(xs):
    return np.divide(1.0, xs)

def morloc_vec_ln(xs):
    return np.log(xs)


# Elementwise conversion to Real (float64).
def morloc_vec_toReal(xs):
    return np.asarray(xs, dtype=np.float64)
