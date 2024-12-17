class HasIdMixin:
    _counter = 0

    def __init__(self, target_cls=None):
        cls = self.__class__ if target_cls is None else target_cls
        cls._counter += 1
        self.id = cls._counter
