from collections.abc import MutableSet
from typing import Any
from weakref import WeakValueDictionary


class WeakSet(MutableSet):
    """
    A Set that uses weak references and does not check for equality.

    Normal sets check that two elements are equal by comparing __hash__
    and __eq__. For tensors, __eq__ is slow so this class only checks
    __hash__. This is a bad idea normally but because we control the hash
    for Tensors, we can (hopefully) use it for gradient calculation

    To avoid circular dependencies, we implement this as a weak set so that
    the garbage collector can clean up objects that are referred to by this
    class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = WeakValueDictionary()

    def __contains__(self, x: Any) -> bool:
        return hash(x) in self._dict

    def __iter__(self):
        return self._dict.values()

    def __len__(self):
        return len(self._dict)

    def add(self, x: Any):
        self._dict[hash(x)] = x

    def discard(self, x: Any):
        if hash(x) in self._dict:
            del self._dict[hash(x)]
