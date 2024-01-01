import math

import numpy as np

from tricycle.ops import (add, cos, div, einsum, exp, log, matmul, max, min, mul, negate, nothing, pow, reduce_sum, sin, sqrt, sub)


def test_can_negate():
    a = np.array([1, 2, 3])
    result = np.array([-1, -2, -3])

    assert np.allclose(negate(a), result)


def test_can_do_nothing():
    a = np.array([1, 2, 3])
    assert np.allclose(nothing(a), a)


def test_can_add():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = np.array([5, 7, 9])
    assert np.allclose(add(a, b), result)


def test_can_subtract():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = np.array([-3, -3, -3])
    assert np.allclose(sub(a, b), result)


def test_can_multiply():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    multiply_2 = np.array([2, 4, 6])
    multiply_minus_1 = np.array([-1, -2, -3])
    multiply_half = np.array([0.5, 1, 1.5])
    multiply_array = np.array([4, 10, 18])

    assert np.allclose(mul(a, np.array(2)), multiply_2)
    assert np.allclose(mul(a, np.array(-1)), multiply_minus_1)
    assert np.allclose(mul(a, np.array(0.5)), multiply_half)
    assert np.allclose(mul(a, b), multiply_array)


def test_can_divide():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    result = np.array([0.25, 0.4, 0.5])
    assert np.allclose(div(a, b), result)


def test_can_reduce_sum():
    a = np.array([1, 2, 3])
    assert np.allclose(reduce_sum(a), 6)


def test_can_exp():
    a = np.array([1, 2, 3])

    result = np.array([math.exp(1), math.exp(2), math.exp(3)])
    assert np.allclose(exp(a), result)


def test_can_log():
    a = np.array([1, 2, 3])

    result = np.array([math.log(1), math.log(2), math.log(3)])
    assert np.allclose(log(a), result)


def test_can_sqrt():
    a = np.array([1, 2, 3])

    result = np.array([math.sqrt(1), math.sqrt(2), math.sqrt(3)])
    assert np.allclose(sqrt(a), result)


def test_can_sin():
    a = np.array([1, 2, math.pi])
    b = np.array([math.sin(1), math.sin(2), 0])
    assert np.allclose(sin(a), b)


def test_can_cos():
    a = np.array([1, 2, math.pi])
    b = np.array([math.cos(1), math.cos(2), -1])
    assert np.allclose(cos(a), b)


def test_can_power():
    a = np.array([1, 2, 3], dtype=float)
    b = np.array([4, 5, 6])

    power_2 = np.array([1, 4, 9])
    power_minus_1 = np.array([1, 1 / 2, 1 / 3])
    power_half = np.array([1, math.sqrt(2), math.sqrt(3)])
    power_b = np.array([1, 32, 729])

    assert np.allclose(pow(a, np.array(2)), power_2)
    assert np.allclose(pow(a, np.array(-1)), power_minus_1)
    assert np.allclose(pow(a, np.array(0.5)), power_half)
    assert np.allclose(pow(a, b), power_b)


def test_can_max():
    a = np.array([1, 2, 3])
    assert np.allclose(max(a), 3)


def test_can_min():
    a = np.array([1, 2, 3])
    assert np.allclose(min(a), 1)


def test_can_matrix_multiply():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.allclose(matmul(a, b), 32)


def test_can_einsum():
    a = np.arange(25).reshape(5, 5)
    b = np.arange(25).reshape(5, 5)
    v = np.arange(5)

    answer = np.einsum("ij,jk->ik", a, b)
    assert np.allclose(einsum(a, b, subscripts="ij,jk->ik"), answer)

    answer = np.einsum("i,i->", v, v)
    assert np.allclose(einsum(v, v, subscripts="i,i->"), answer)

    answer = np.einsum("i->", v)
    assert np.allclose(einsum(v, subscripts="i->"), answer)

    answer = np.einsum("ij->j", a)
    assert np.allclose(einsum(a, subscripts="ij->j"), answer)

    answer = np.einsum("ij,j->i", a, v)
    assert np.allclose(einsum(a, v, subscripts="ij,j->i"), answer)
