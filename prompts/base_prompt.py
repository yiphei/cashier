class CallableMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        return (
            instance.f_string_prompt.format(**kwargs)
            if instance.f_string_prompt
            else instance.dynamic_prompt(**kwargs)
        )


class Prompt(metaclass=CallableMeta):
    f_string_prompt = None

    def __init__(self, **kwargs):
        pass

    def dynamic_prompt(self, **kwargs):
        return None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        has_fstring = cls.f_string_prompt is not None
        has_dynamic = cls.dynamic_prompt != Prompt.dynamic_prompt

        if not has_fstring and not has_dynamic:
            raise NotImplementedError(
                f"Class {cls.__name__} must override either f_string_prompt or dynamic_prompt"
            )
        if has_fstring and has_dynamic:
            raise ValueError(
                f"Class {cls.__name__} should not override both f_string_prompt and dynamic_prompt"
            )
