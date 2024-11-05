from prompts.base_prompt import BasePrompt
from prompts.cashier_background import CashierBackgroundPrompt


class NodeSystemPrompt(BasePrompt):

    def dynamic_prompt(
        self, node_prompt, input, node_input_json_schema, state_json_schema, last_msg
    ):
        NODE_PROMPT = (
            CashierBackgroundPrompt.f_string_prompt + "\n\n"
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
            "<response_guidelines>\n"
            "- because your responses will be converted to speech, "
            "you must respond in a conversational way: natural, easy to understand when converted to speech, and generally concise and brief (no long responses).\n"
            "- AVOID using any rich text formatting like hashtags, bold, italic, bullet points, numbered points, headers, etc.\n"
            "- When responding to customers, AVOID providing unrequested information.\n"
            "- If a response to a request is naturally long, then either ask claryfing questions to further refine the request, "
            "summarize the response, or break down the response in many separate responses.\n"
            "- Overall, try to be professional, polite, empathetic, and friendly\n"
            "</response_guidelines>\n"
            "<state_guidelines>\n"
            "- Among the tools provided, there are functions for getting and updating the state defined in <state>. "
            "For state updates, you will have field specific update functions, whose names are `update_state_<field>` and where <field> is a state field.\n"
            "- You must update the state whenever applicable and as soon as possible. You cannot proceed to the next stage of the conversation without updating the state\n"
            "- Only you can update the state, so there is no need to udpate the state to the same value that had already been updated to in the past.\n"
            + (
                "- state updates can only happen in response to new messages (i.e. messages after <cutoff_msg>).\n"
                if last_msg
                else ""
            )
            + "</state_guidelines>\n"
            "<tools_guidelines>\n"
            "- Minimize reliance on external knowledge. Always retrieve information from the system prompts and available tools. "
            "If they dont provide the information needed, just say you do not know.\n"
            "- AVOID stating/mentioning that you can/will perform an action if there are no tools (including state updates) associated with that action.\n"
            "- if you need to perform an action, you can only state to the customer that you performed it after the associated tool (including state update) calls have been successfull.\n"
            "</tools_guidelines>\n"
            "<general_guidelines>\n"
            "- think step-by-step before you respond.\n"
            "- you must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
            + (
                "- everthing stated in <instructions> and here in <guidelines> only applies to the conversation starting after <cutoff_msg>\n"
                if last_msg
                else ""
            )
            + "</general_guidelines>\n"
            "</guidelines>"
        )

        NODE_PROMPT += GUIDELINES
        kwargs = {"state_json_schema": state_json_schema}
        if input is not None:
            kwargs["node_input"] = input
            kwargs["node_input_json_schema"] = node_input_json_schema

        return NODE_PROMPT.format(**kwargs)
