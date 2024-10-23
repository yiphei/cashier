from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from db_functions import Order
from model import AssistantTurn
from model_tool_decorator import (
    get_anthropic_tool_def_from_oai,
    get_oai_tool_def_from_fields,
)
from model_util import ModelProvider

BACKGROUND = (
    "You are a cashier working for the coffee shop Heaven Coffee. You are physically embedded inside the shop, "
    "so you will interact with real in-person customers. There is a microphone that transcribes customer's speech to text, "
    "and a speaker that outputs your text to speech."
)


class NodeSchema:
    _counter = 0
    OPENAI_TOOL_NAME_TO_TOOL_DEF = {}
    ANTHROPIC_TOOL_NAME_TO_TOOL_DEF = {}

    model_provider_to_tool_def = {
        ModelProvider.OPENAI: OPENAI_TOOL_NAME_TO_TOOL_DEF,
        ModelProvider.ANTHROPIC: ANTHROPIC_TOOL_NAME_TO_TOOL_DEF,
    }

    def __init__(
        self,
        node_prompt,
        tool_fn_names,
        input_pydantic_model,
        state_pydantic_model,
        first_turn=None,
    ):
        NodeSchema._counter += 1
        self.id = NodeSchema._counter
        self.node_prompt = node_prompt
        self.tool_fn_names = tool_fn_names
        self.is_initialized = False
        self.input_pydantic_model = input_pydantic_model
        self.state_pydantic_model = state_pydantic_model
        self.first_turn = first_turn

        for field_name, field_info in self.state_pydantic_model.model_fields.items():
            new_tool_fn_name = f"update_state_{field_name}"
            field_args = {field_name: (field_info.annotation, field_info)}
            update_state_fn_json_schema = get_oai_tool_def_from_fields(
                new_tool_fn_name,
                f"Function to update the `{field_name}` field in the state",
                field_args,
            )
            self.OPENAI_TOOL_NAME_TO_TOOL_DEF[new_tool_fn_name] = (
                update_state_fn_json_schema
            )
            self.ANTHROPIC_TOOL_NAME_TO_TOOL_DEF[new_tool_fn_name] = (
                get_anthropic_tool_def_from_oai(update_state_fn_json_schema)
            )
            self.tool_fn_names.append(new_tool_fn_name)

        get_state_oai_tool_def = get_oai_tool_def_from_fields(
            "get_state", "Function to get the current state", {}
        )
        get_state_anthropic_tool_def = get_anthropic_tool_def_from_oai(
            get_state_oai_tool_def
        )
        self.OPENAI_TOOL_NAME_TO_TOOL_DEF["get_state"] = get_state_oai_tool_def
        self.ANTHROPIC_TOOL_NAME_TO_TOOL_DEF["get_state"] = get_state_anthropic_tool_def
        self.tool_fn_names.append("get_state")

    def run(self, input):
        self.is_initialized = True
        if input is not None:
            assert isinstance(input, self.input_pydantic_model)

        self.state = self.state_pydantic_model()
        self.prompt = self.generate_system_prompt(
            self.input_pydantic_model is not None,
            node_prompt=self.node_prompt,
            node_input=(
                input.model_dump_json()
                if self.input_pydantic_model is not None
                else None
            ),
            node_input_json_schema=(
                self.input_pydantic_model.model_json_schema()
                if self.input_pydantic_model is not None
                else None
            ),
        )

    def generate_system_prompt(self, has_input, **kwargs):
        NODE_PROMPT = (
            BACKGROUND + "\n\n"
            "# EXPECTATION\n\n"
            "<!--- This section describes what the conversation will be about and what you are expected to do --->\n"
            "{node_prompt}\n\n"
            "# GUIDELINES\n\n"
            "<!--- This section enumerates important guidelines on how you should behave. These must be strictly followed --->\n"
            "## Response\n\n"
            "- because your responses will be converted to speech, "
            "you must respond in a conversational way: natural, easy to understand when converted to speech, and generally concise and brief (no long responses).\n"
            "- DO NOT use any rich text formatting like hashtags, bold, italic, bullet points, numbered points, headers, etc.\n"
            "- When responding to customers, DO NOT provide unrequested information.\n"
            "- If a response to a request is naturally long, then either ask claryfing questions to further refine the request, "
            "summarize the response, or break down the response in many separate responses.\n"
            "- Overall, try to be professional, polite, empathetic, and friendly\n\n"
            "## Tools and state\n\n"
            "- Minimize reliance on external knowledge. Always retrieve information from the prompts and tools. "
            "If they dont provide the information you need, just say you do not know.\n"
            "- Among the tools provided, there are state update functions, whose names start with `update_state_<field>` and where <field> is a state field. You must update the state object whenever applicable "
            "and as soon as possible. You cannot proceed to the next stage of the conversation without updating the state.\n"
            # "- you must not assume that tools and state updates used before this message are still available.\n"
            "- you must not state/mention that you can/will perform an action if there are no tools (including state updates) associated with that action.\n"
            "- if you need to perform an action, you can only state to the customer that you performed it after the associated tool (including state update) calls have been successfull.\n"
            "- state updates can only happen in response to new messages, not messages prior to this message\n\n"
            # "- you must use tools whenever possible and as soon as possible. "
            # "This is because there usually is an associated tool for every user input and that tool will help you with the user input. "
            # "When in doubt, use the tools.\n"
            "## General\n\n"
            "- think very hard before you respond.\n"
            "- if there are messages before this message, consider those as part of the current conversation but treat them as references only.\n"
            "- you must decline to do anything that is not explicitly covered by the EXPECTATION and IMPORTANT NOTES section.\n\n"
        )
        if has_input:
            NODE_PROMPT += (
                "# INPUT\n\n"
                "<!--- This section provides the input to the conversation. The input contains valuable information that help you accomplish the expectation stated above. "
                "You will be provided both the input (in JSON format) and the input's JSON schema --->\n"
                "INPUT:\n"
                "```\n"
                "{node_input}\n"
                "```\n"
                "INPUT JSON SCHEMA:\n"
                "```\n"
                "{node_input_json_schema}\n"
                "```\n\n"
            )

        # NODE_PROMPT += (
        #     "# IMPORTANT NOTES\n\n"
        #     "Treat these notes with the utmost importance:\n"
        #     "- because your responses will be converted to speech, "
        #     "you must respond in a conversational way: natural, easy to understand when converted to speech, and generally concise and brief\n"
        #     "- DO NOT use any rich text formatting like hashtags, bold, italic, bullet points, numbered points, headers, etc.\n"
        #     "- When responding to customers, DO NOT provide unrequested information.\n"
        #     "- If a response to a request is naturally long, then either ask claryfing questions to further refine the request, "
        #     "summarize the response, or break down the response in many separate responses.\n"
        #     "- Minimize reliance on external knowledge. Always get information from the prompts and tools."
        #     "If they dont provide the information you need, just say you do not know.\n"
        #     "- Overall, be professional, polite, empathetic, and friendly.\n"
        #     "- if there are messages before this stage, use them as a reference ONLY.\n"
        #     "- you must decline to do anything that is not explicitly covered by EXPECTATION or BACKGROUND.\n"
        #     "- you must not assume that tools and state updates used previously are still available.\n"
        #     "- you must not state/mention that you can/will perform an action if there are no tools or state updates associated with that action.\n"
        #     "- if you need to perform an action, you can only state to the customer that you performed it after the associated tool and/or state update calls have been successfull.\n"
        #     "- you must update the state object whenever possible. "
        #     "There is a specific update tool for each state field, and all state update tool names start with `update_state_*`. "
        #     "You cannot proceed to the next stage without updating the state.\n"
        #     "- state updates should only happen in response to new messages, not messages prior to this stage"
        #     "- you must use tools whenever possible and as soon as possible. "
        #     "This is because there usually is an associated tool for every user input and that tool will help you with the user input. "
        #     "When in doubt, use the tools.\n"
        # )

        return NODE_PROMPT.format(**kwargs)

    def update_state(self, **kwargs):
        old_state = self.state.model_dump()
        new_state = old_state | kwargs
        self.state = self.state_pydantic_model(**new_state)

    def get_state(self):
        return self.state


class EdgeSchema:
    _counter = 0

    def __init__(
        self,
        from_node_schema,
        to_node_schema,
        state_condition_fn,
        new_input_from_state_fn,
    ):
        EdgeSchema._counter += 1
        self.id = EdgeSchema._counter
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
        self.new_input_from_state_fn = new_input_from_state_fn

    def check_state_condition(self, state):
        return self.state_condition_fn(state)


## Chain ##


class BaseStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TakeOrderState(BaseStateModel):
    order: Optional[Order] = None
    has_finished_ordering: bool = Field(
        description=(
            "whether the customer has finished ordering. This can only be true after"
            " you have explicitly confirmed with them that they have finished ordering,"
            " by asking questions like 'Anything else?'."
        ),
        default=False,
    )


take_order_node_schema = NodeSchema(
    node_prompt=(
        "First, greet the customer. Then, your main job is to take their orders, which"
        " also includes answering reasonable questions about the shop & menu only and"
        " assisting them with any issues they may have about their orders. Things you"
        " must keep in mind:\n"
        "- never assume the drink size\n"
        "- given a size, always assume the default option values of that size, unless the customer"
        " explicitly specifies otherwise. This also means that you do not need to ask if they are ok with the default option values,"
        " and you do not need to spell them out (not even say e.g. 'grande size with the default options'), unless the customer asks otherwise."
        " But if an option does not have a default value, do not make any assumptions.\n"
        "- as soon as a single drink order is completed (e.g. all the required options are specified),"
        " you must restate the order back to the customer to ensure that it is correct. For the option values,"
        " only state those that were explicitly specified by the customer, so do not state default options"
        " assumed for the size but do state any other option, even default option values that were specified by the customer.\n\n"
        "Lastly, if they are not immediately ready to order after the greeting, you can also engage in some"
        " small talk about any topic, but you need to steer the conversation back to ordering after some"
        " back-and-forths."
    ),
    tool_fn_names=[
        "get_menu_items_options",
        "get_menu_item_from_name",
    ],
    input_pydantic_model=None,
    state_pydantic_model=TakeOrderState,
    first_turn=AssistantTurn(
        msg_content="hi, welcome to Heaven Coffee", model_provider=ModelProvider.NONE
    ),
)


class ConfirmOrderState(BaseStateModel):
    has_confirmed_order: bool = Field(
        description="whether the customer has confirmed their order", default=False
    )


confirm_order_node_schema = NodeSchema(
    node_prompt=(
        "Confirm the order with the customer. You do this by"
        " repeating the order back to them and get their confirmation."
    ),
    tool_fn_names=[],
    input_pydantic_model=Order,
    state_pydantic_model=ConfirmOrderState,
)

take_to_confirm_edge_schema = EdgeSchema(
    from_node_schema=take_order_node_schema,
    to_node_schema=confirm_order_node_schema,
    state_condition_fn=lambda state: state.has_finished_ordering
    and state.order is not None,
    new_input_from_state_fn=lambda state: state.order,
)


class TerminalOrderState(BaseStateModel):
    has_said_goodbye: bool = Field(
        description="whether the customer has said goodbye", default=False
    )


terminal_order_node_schema = NodeSchema(
    node_prompt=("Order has been successfully placed. Thank the customer."),
    tool_fn_names=[],
    input_pydantic_model=None,
    state_pydantic_model=TerminalOrderState,
)

confirm_to_terminal_edge_schema = EdgeSchema(
    from_node_schema=confirm_order_node_schema,
    to_node_schema=terminal_order_node_schema,
    state_condition_fn=lambda state: state.has_confirmed_order,
    new_input_from_state_fn=lambda state: None,
)

FROM_NODE_ID_TO_EDGE_SCHEMA = {
    take_order_node_schema.id: [take_to_confirm_edge_schema],
    confirm_order_node_schema.id: [confirm_to_terminal_edge_schema],
}
