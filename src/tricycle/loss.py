from abc import ABC, abstractmethod

from tricycle.binary import bmul
from tricycle.ops import softmax
from tricycle.reduce import radd
from tricycle.tensor import Tensor
from tricycle.unary import ulog, umul


class LossFn(ABC):
    @abstractmethod
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vectorise(self) -> "VectorisedLossFn":
        raise NotImplementedError


class VectorisedLossFn(LossFn):
    def vectorise(self) -> "VectorisedLossFn":
        return self

    @abstractmethod
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError


class VectorisedMeanSquareError(VectorisedLossFn):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        square_error = (y_true - y_pred) ** 2
        return radd(square_error, "ki->k")


class MeanSquareError(LossFn):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        square_error = (y_true - y_pred) ** 2
        return radd(square_error, "i->")

    def vectorise(self) -> "VectorisedLossFn":
        return VectorisedMeanSquareError()


class VectorisedCrossEntropy(VectorisedLossFn):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # normalise and log
        y_pred = ulog(softmax(y_pred))
        return umul(radd(bmul(y_true, y_pred), "ki->k"), -1)


class CrossEntropy(LossFn):
    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the cross entropy loss
        """
        # normalise and log
        y_pred = ulog(softmax(y_pred))
        return umul(radd(bmul(y_true, y_pred), "i->"), -1)

    def vectorise(self) -> "VectorisedLossFn":
        return VectorisedCrossEntropy()
