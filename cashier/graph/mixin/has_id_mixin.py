class HasIdMixin:
    _counter = 0

    def __init__(self):
        HasIdMixin._counter += 1
        self.id = HasIdMixin._counter
