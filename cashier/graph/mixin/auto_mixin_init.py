from inspect import signature


class AutoMixinInit(type):
    """Metaclass that automatically initializes mixins in the correct order."""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)

        # Get all base classes that end with 'Mixin'
        mixins = [base for base in cls.__bases__ if base.__name__.endswith("Mixin")]
        first_base = next(
            (base for base in cls.__bases__ if not base.__name__.endswith("Mixin")),
            None,
        )

        # Initialize each mixin with matching kwargs
        for mixin in mixins:
            # Get the init parameters for this mixin
            if hasattr(mixin, "__init__"):
                # Get only the parameter names from the function signature
                init_params = list(signature(mixin.__init__).parameters.keys())[
                    1:
                ]  # Skip 'self'

                # Filter kwargs to only include parameters that match this mixin's init
                mixin_kwargs = {k: v for k, v in kwargs.items() if k in init_params}

                # Call the mixin's init
                mixin.__init__(instance, **mixin_kwargs)

        if "__init__" in cls.__dict__:
            cls.__init__(instance, *args, **kwargs)
        elif first_base is not None:
            first_base.__init__(instance, **kwargs)
        return instance
