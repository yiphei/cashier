class Ref:
    def __init__(self, obj, attr):
        self._obj = obj
        self._attr = attr

    def __get__(self, instance, owner):
        return getattr(self._obj, self._attr)

    def __set__(self, instance, value):
        setattr(self._obj, self._attr, value)

    # Optional: make it behave more like a regular variable
    def __repr__(self):
        return repr(self.__get__(None, None))

    def __getattr__(self, name):
        # Get the current value and access the requested attribute
        value = self.__get__(None, None)
        return getattr(value, name)