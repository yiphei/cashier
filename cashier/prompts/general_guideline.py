from cashier.prompts.base_prompt import BasePrompt


class GeneralGuidelinePrompt(BasePrompt):

    def dynamic_prompt(
        self,
        last_msg,
    ):
        return (
            "<general_guidelines>\n"
            "- think step-by-step before you respond.\n"
            "- you must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
            + (
                "- everthing stated in <instructions> and here in <guidelines> only applies to the conversation starting after <cutoff_msg>\n"
                if last_msg
                else ""
            )
            + "</general_guidelines>\n"
        )
