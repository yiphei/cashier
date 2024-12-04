class HasIdMixin:
    _counter = 0

    def __init__(self):
        self.__class__._counter += 1
        self.id = self.__class__._counter
