class Prompt:
    f_string_prompt = None

    def dynamic_prompt(self, **kwargs):
        return None

    def __call__(self, **kwargs):
        return (
            self.f_string_prompt.format(**kwargs)
            if self.f_string_prompt
            else self.dynamic_prompt(**kwargs)
        )

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
