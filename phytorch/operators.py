
import math
from typing import Callable, Iterable

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    return 1 if x < y else 0

def eq(x: float, y: float) -> float:
    return 1 if x == y else 0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float, eps: float = 1e-2) -> float:
    return 1 if abs(x - y) < eps else 0

def sigmoid(x: float) -> float:
    return (1 / (1 + math.e**(-x))) if x > 0 else (1 / (1 + math.e**(x)))

def relu(x: float) -> float:
    return x if x > 0 else 0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1 / x

def log_back(x: float, d: float) -> float:
    return d * (1 / x)

def inv_back(x: float, d: float) -> float:
    return d * -(1 / x**2)

def relu_back(x: float, d: float) -> float:
    return d * (1 if x > 0 else 0)

