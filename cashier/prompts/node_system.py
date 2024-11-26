from typing import Any, Dict, Optional, Type

from cashier.prompts.base_prompt import BasePrompt
from cashier.prompts.general_guideline import GeneralGuidelinePrompt
from cashier.prompts.response_guideline import ResponseGuidelinePrompt
from cashier.prompts.state_guideline import StateGuidelinePrompt
from cashier.prompts.tool_guideline import ToolGuidelinePrompt


class NodeSystemPrompt(BasePrompt):
    BACKGROUND_PROMPT: Optional[Type[BasePrompt]] = None
    GUIDELINE_PROMPTS = [
        ResponseGuidelinePrompt,
        StateGuidelinePrompt,
        ToolGuidelinePrompt,
        GeneralGuidelinePrompt,
    ]

    def dynamic_prompt(  # type: ignore
        self,
        node_prompt: str,
        input: Any,
        node_input_json_schema: Optional[Dict],
        curr_request: str,
        state_json_schema: Optional[Dict],
        last_msg: Optional[str],
    ) -> str:
        fn_kwargs = locals()
        fn_kwargs.pop("self")
        NODE_PROMPT = (
            self.BACKGROUND_PROMPT() + "\n\n"  # type: ignore
            "This request section describes what the overall customer request is\n"
            "<request>\n"
            f"{curr_request}\n"
            "</request>\n\n"
            "This instructions section describes what the current conversation is supposed to be about and what you are expected to do. The instructions only address a single part of the overall customer request.\n"
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

        if state_json_schema is not None:
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
            GUIDELINES += guideline(strict_kwargs_check=False, **fn_kwargs)  # type: ignore

        GUIDELINES += "</guidelines>"

        NODE_PROMPT += GUIDELINES
        kwargs = {}
        if state_json_schema is not None:
            kwargs["state_json_schema"] = state_json_schema
        if input is not None:
            kwargs["node_input"] = input
            kwargs["node_input_json_schema"] = node_input_json_schema

        return NODE_PROMPT.format(**kwargs)
