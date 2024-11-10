from cashier.prompts.base_prompt import BasePrompt
from cashier.prompts.general_guideline import GeneralGuidelinePrompt
from cashier.prompts.state_guideline import StateGuidelinePrompt
from cashier.prompts.response_guideline import ResponseGuidelinePrompt
from cashier.prompts.tool_guideline import ToolGuidelinePrompt

class NodeSystemPrompt(BasePrompt):
    GUIDELINE_PROMPTS = [ResponseGuidelinePrompt, StateGuidelinePrompt, ToolGuidelinePrompt, GeneralGuidelinePrompt]

    def dynamic_prompt(
        self,
        node_prompt,
        background_prompt,
        input,
        node_input_json_schema,
        state_json_schema,
        last_msg,
    ):
        fn_kwargs = locals()
        fn_kwargs.pop('self')
        NODE_PROMPT = (
            background_prompt + "\n\n"
            "This instructions section describes what the conversation is supposed to be about and what you are expected to do\n"
            "<instructions>\n"
            f"{node_prompt}\n"
            "</instructions>\n\n"
        )
        if input is not None:
            NODE_PROMPT += (
                "This section provides the input to the conversation. The input contains valuable information that help you accomplish the instructions in <instructions>. "
                "You will be provided with both the input (in JSON format) and its JSON schema\n"
                "<input>\n"
                "<input_json>\n"
                "{node_input}\n"
                "</input_json>\n"
                "<input_json_schema>\n"
                "{node_input_json_schema}\n"
                "</input_json_schema>\n"
                "</input>\n\n"
            )

        NODE_PROMPT += (
            "This section provides the state's json schema. The state keeps track of important data during the conversation.\n"
            "<state>\n"
            "{state_json_schema}\n"
            "</state>\n\n"
        )

        if last_msg:
            NODE_PROMPT += (
                "This is the cutoff message. Everything stated here only applies to messages after the cutoff message. All messages until the cutoff message represent a historical conversation "
                "that you may use as a reference.\n"
                "<cutoff_msg>\n"  # can explore if it's better to have two tags: cutoff_customer_msg and cutoff_assistant_msg
                f"{last_msg}\n"
                "</cutoff_msg>\n\n"
            )

        GUIDELINES = (
            "This guidelines section enumerates important guidelines on how you should behave. These must be strictly followed\n"
            "<guidelines>\n"
        )
        for guideline in self.GUIDELINE_PROMPTS:
            GUIDELINES += guideline(strict_kwargs_check=False, **fn_kwargs)

        GUIDELINES += (
            "</guidelines>"
        )

        NODE_PROMPT += GUIDELINES
        kwargs = {"state_json_schema": state_json_schema}
        if input is not None:
            kwargs["node_input"] = input
            kwargs["node_input_json_schema"] = node_input_json_schema

        return NODE_PROMPT.format(**kwargs)
