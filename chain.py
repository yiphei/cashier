import copy
from enum import StrEnum
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
        self.input_pydantic_model = input_pydantic_model
        self.state_pydantic_model = state_pydantic_model
        self.first_turn = first_turn
        self.tool_registry = ToolRegistry()

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
            "get_state", "Function to get the current state, as defined in <state>", {}
        )
        self.tool_fn_names.append("get_state")

    def create_node(self, input, last_msg=None, prev_node=None, edge_schema=None):
        if input is not None:
            assert isinstance(input, self.input_pydantic_model)
            assert prev_node is None
        elif prev_node is not None:
            assert input is None
            input = prev_node.input

        if prev_node is None:
            state = self.state_pydantic_model()
        else:
            state = prev_node.state.copy_reset()

        prompt = self.generate_system_prompt(
            (
                input.model_dump_json()
                if self.input_pydantic_model is not None
                else None
            ),
            last_msg,
        )

        direction = (
            Node.Direction.BWD
            if edge_schema and edge_schema.from_node_schema == self
            else Node.Direction.FWD
        )
        return Node(self, input, state, prompt, edge_schema, direction)

    def generate_system_prompt(self, input, last_msg):
        NODE_PROMPT = (
            BACKGROUND + "\n\n"
            "This instructions section describes what the conversation is supposed to be about and what you are expected to do\n"
            "<instructions>\n"
            f"{self.node_prompt}\n"
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
        kwargs = {"state_json_schema": self.state_pydantic_model.model_json_schema()}
        if input is not None:
            kwargs["node_input"] = input
            kwargs["node_input_json_schema"] = (
                self.input_pydantic_model.model_json_schema()
            )

        return NODE_PROMPT.format(**kwargs)


class Node:
    _counter = 0

    class Status(StrEnum):
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"

    class Direction(StrEnum):
        FWD = "FWD"
        BWD = "BWD"

    def __init__(self, schema, input, state, prompt, edge_schema, direction):
        Node._counter += 1
        self.id = Node._counter
        self.state = state
        self.prompt = prompt
        self.input = input
        self.schema = schema
        self.first_user_message = False
        self.status = self.Status.IN_PROGRESS
        self.edge_schema = edge_schema
        self.direction = direction

    def mark_as_completed(self):
        self.status = self.Status.COMPLETED

    def update_state(self, **kwargs):
        if self.first_user_message:
            old_state = self.state.model_dump()
            new_state = old_state | kwargs
            self.state = self.state.__class__(**new_state)
        else:
            raise StateUpdateError(
                "cannot update any state field until first customer message"
            )

    def get_state(self):
        return self.state

    def update_first_user_message(self):
        self.first_user_message = True


class BwdTransType(StrEnum):
    RESET = "RESET"
    KEEP = "KEEP"
    KEEP_IF_INPUT_UNCHANGED = "KEEP_IF_INPUT_UNCHANGED"


class FwdSkipType(StrEnum):
    SKIP = "SKIP"
    SKIP_IF_INPUT_UNCHANGED = "SKIP_IF_INPUT_UNCHANGED"


class EdgeSchema:
    _counter = 0

    def __init__(
        self,
        from_node_schema,
        to_node_schema,
        state_condition_fn,
        new_input_from_state_fn,
        bwd_trans_type=BwdTransType.RESET,
        fwd_from_complete_to_prev_complete=None,
        fwd_from_complete_to_prev_incomplete=None,
        fwd_from_incomplete_to_prev_complete=None,
        fwd_from_incomplete_to_prev_incomplete=None,
    ):
        EdgeSchema._counter += 1
        self.id = EdgeSchema._counter
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
        self.new_input_from_state_fn = new_input_from_state_fn
        self.bwd_trans_type = bwd_trans_type
        self.fwd_from_complete_to_prev_complete = fwd_from_complete_to_prev_complete
        self.fwd_from_complete_to_prev_incomplete = fwd_from_complete_to_prev_incomplete
        # these two below assume that it was previously completed
        self.fwd_from_incomplete_to_prev_complete = fwd_from_incomplete_to_prev_complete
        self.fwd_from_incomplete_to_prev_incomplete = (
            fwd_from_incomplete_to_prev_incomplete
        )

    def check_state_condition(self, state):
        return self.state_condition_fn(state)


## Chain ##


class BaseStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def copy_reset(self):
        new_data = copy.deepcopy(dict(self))

        # Iterate through fields and reset those marked as resettable
        for field_name, field_info in self.model_fields.items():
            # Check if field has the resettable marker in its metadata
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "resettable"
            ):
                new_data[field_name] = field_info.default

        return self.__class__(**new_data)


class TakeOrderState(BaseStateModel):
    order: Optional[Order] = None
    has_finished_ordering: bool = Field(
        description=(
            "whether the customer has finished ordering. This can only be true after"
            " you have explicitly confirmed with them that they have finished ordering,"
            " by asking questions like 'Anything else?'."
        ),
        default=False,
        resettable=True,
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
        description="whether the customer has confirmed their order",
        default=False,
        resettable=True,
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
        description="whether the customer has said goodbye",
        default=False,
        resettable=True,
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

TO_NODE_ID_TO_EDGE_SCHEMA = {
    terminal_order_node_schema.id: [confirm_to_terminal_edge_schema],
    confirm_order_node_schema.id: [take_to_confirm_edge_schema],
}
