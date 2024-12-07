from inspect import signature

class AutoMixinInit(type):
    """Metaclass that automatically initializes mixins in the correct order."""

    def _initialize_mixins(cls, instance, args, kwargs):
        # Get all base classes that end with 'Mixin'
        mixins = [base for base in cls.__mro__ if base.__name__.endswith("Mixin")]

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

    def __new__(mcs, name, bases, attrs):
        original_init = attrs.get('__init__')
        
        def wrapped_init(self, *args, **kwargs):
            # Call the mixin initialization
            self.__class__._initialize_mixins(self, args, kwargs)
            # Call original init if it exists
            if original_init:
                original_init(self, *args, **kwargs)
            else:
                super(self.__class__, self).__init__(*args, **kwargs)
        
        attrs['__init__'] = wrapped_init
        return super().__new__(mcs, name, bases, attrs)