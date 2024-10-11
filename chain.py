from typing import Optional

from openai import pydantic_function_tool
from pydantic import BaseModel, Field, create_model

from db_functions import OPENAI_TOOL_NAME_TO_TOOL_DEF, Order


class NodeSchema:
    _counter = 0

    def __init__(
        self, node_prompt, tool_fns, input_pydantic_model, state_pydantic_model
    ):
        NodeSchema._counter += 1
        self.id = NodeSchema._counter
        self.node_prompt = node_prompt
        self.tool_fns = tool_fns
        self.is_initialized = False
        self.input_pydantic_model = input_pydantic_model
        self.state_pydantic_model = state_pydantic_model

        def remove_default(schema):
            found_key = False
            for key, value in schema.items():
                if key == "default":
                    found_key = True
                elif isinstance(value, dict):
                    remove_default(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            remove_default(item)
            if found_key:
                schema.pop("default")

        for field_name, field_info in self.state_pydantic_model.model_fields.items():
            field_args = {field_name: (field_info.annotation, field_info)}
            fn_pydantic_model = create_model(f"update_state_{field_name}", **field_args)
            update_state_json_schema = pydantic_function_tool(
                fn_pydantic_model,
                name=f"update_state_{field_name}",
                description=f"Function to update the `{field_name}` field in the state",
            )
            remove_default(update_state_json_schema)
            self.tool_fns.append(update_state_json_schema)

        self.tool_fns.append(
            {
                "type": "function",
                "function": {
                    "name": "get_state",
                    "strict": True,
                    "description": "Function to get the current state",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        )

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
        )

    def generate_system_prompt(self, has_input, **kwargs):
        NODE_PROMPT = (
            "You are now in the next stage of the conversation. The description and main expectation for this stage is the following:\n"
            "```\n"
            "{node_prompt}\n"
            "```\n\n"
        )
        if has_input:
            NODE_PROMPT += (
                "There is an input to this stage. It contains "
                "valuable information that help you accomplish the main expectation. The input is the following in JSON format:\n"
                "```\n"
                "{node_input}\n"
                "```\n\n"
            )

        NODE_PROMPT += (
            "During this stage, you must use function calls whenever possible and as soon as possible. "
            "This is because there usually is an associated function for every user input and that function will help you with the user input. "
            "When in doubt, use the function/s. In conjunction, you must update the state object whenever possible. "
            "The state update function is `update_state_*` and getting the current state function is `get_state`. "
            "You cannot proceed to the next stage without updating the state."
        )

        return NODE_PROMPT.format(**kwargs)

    def update_state(self, **kwargs):
        old_state = self.state.model_dump()
        new_state = old_state | kwargs
        self.state = self.state_pydantic_model(**new_state)

    def get_state(self):
        return self.state.model_dump_json()


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


class TakeOrderState(BaseModel):
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
        " assisting them with any issues they may have about their orders. If they are"
        " not immediately ready to order after the greeting, you may engage in some"
        " small talk but you need to steer the conversation back to ordering after 4"
        " back-and-forths."
    ),
    tool_fns=[
        OPENAI_TOOL_NAME_TO_TOOL_DEF["get_menu_items_options"],
        OPENAI_TOOL_NAME_TO_TOOL_DEF["get_menu_item_from_name"],
    ],
    input_pydantic_model=None,
    state_pydantic_model=TakeOrderState,
)


class ConfirmOrderState(BaseModel):
    has_confirmed_order: bool = Field(
        description="whether the customer has confirmed their order", default=False
    )


confirm_order_node_schema = NodeSchema(
    node_prompt=(
        "Your main job is to confirm the order with the customer. You do this by"
        " repeating the order back to them and get their confirmation."
    ),
    tool_fns=[],
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


class TerminalOrderState(BaseModel):
    has_said_goodbye: bool = Field(
        description="whether the customer has said goodbye", default=False
    )


terminal_order_node_schema = NodeSchema(
    node_prompt=("Order has been successfully placed. Thank the customer."),
    tool_fns=[],
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
