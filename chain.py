from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from db_functions import Order
from function_call_context import StateUpdateError
from model import AssistantTurn
from model_tool_decorator import ToolRegistry
from model_util import ModelProvider

BACKGROUND = (
    "You are a cashier working for the coffee shop Heaven Coffee. You are physically embedded inside the shop, "
    "so you will interact with real in-person customers. There is a microphone that transcribes customer's speech to text, "
    "and a speaker that outputs your text to speech."
)


class NodeSchema:
    _counter = 0

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
        self.tool_registry = ToolRegistry()
        self.first_user_message = False

        for field_name, field_info in self.state_pydantic_model.model_fields.items():
            new_tool_fn_name = f"update_state_{field_name}"
            field_args = {field_name: (field_info.annotation, field_info)}
            self.tool_registry.add_tool_def(
                new_tool_fn_name,
                f"Function to update the `{field_name}` field in the state",
                field_args,
            )
            self.tool_fn_names.append(new_tool_fn_name)

        self.tool_registry.add_tool_def(
            "get_state", "Function to get the current state", {}
        )
        self.tool_fn_names.append("get_state")

    def update_first_user_message(self):
        self.first_user_message = True

    def run(self, input, last_user_msg=None):
        self.is_initialized = True
        if input is not None:
            assert isinstance(input, self.input_pydantic_model)

        self.state = self.state_pydantic_model()
        self.prompt = self.generate_system_prompt(
            self.input_pydantic_model is not None,
            last_user_msg,
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
            state_json_schema=self.state_pydantic_model.model_json_schema(),
        )

    def generate_system_prompt(self, has_input, last_user_msg, **kwargs):
        NODE_PROMPT = (
            BACKGROUND + "\n\n"
            "This instructions section describes what the conversation will be about and what you are expected to do\n"
            "<instructions>\n"
            f"{self.node_prompt}\n"
            "</instructions>\n\n"
        )
        if has_input:
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

        if last_user_msg:
            NODE_PROMPT += (
                "This is the last customer message. All messages until the last customer message represent a historical conversation "
                "that you may use as a reference.\n"
                "<last_customer_msg>\n"
                f"{last_user_msg}\n"
                "</last_customer_msg>\n\n"
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
                "- state updates can only happen in response to new customer messages (i.e. messages after <last_customer_msg>).\n"
                if last_user_msg
                else ""
            )
            + "</state_guidelines>\n"
            "<tools_guidelines>\n"
            "- Minimize reliance on external knowledge. Always retrieve information from the system prompts and available tools. "
            "If they dont provide the information you need, just say you do not know.\n"
            "- AVOID stating/mentioning that you can/will perform an action if there are no tools (including state updates) associated with that action.\n"
            "- if you need to perform an action, you can only state to the customer that you performed it after the associated tool (including state update) calls have been successfull.\n"
            "</tools_guidelines>\n"
            "<general_guidelines>\n"
            "- think very hard before you respond.\n"
            "- you must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
            + (
                "- everthing stated in <instructions> and here in <guidelines> only applies to the conversation starting after <last_customer_msg>\n"
                if last_user_msg
                else ""
            )
            + "</general_guidelines>\n"
            "</guidelines>"
        )

        NODE_PROMPT += GUIDELINES

        return NODE_PROMPT.format(**kwargs)

    def update_state(self, **kwargs):
        if self.first_user_message:
            old_state = self.state.model_dump()
            new_state = old_state | kwargs
            self.state = self.state_pydantic_model(**new_state)
        else:
            raise StateUpdateError("cannot update until first user message")

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
