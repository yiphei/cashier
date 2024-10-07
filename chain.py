from pydantic import BaseModel, Field

from db_functions import Order


class NodeSchema:
    def __init__(
        self, node_prompt, tool_fns, input_pydantic_model, state_pydantic_model
    ):
        self.node_prompt = node_prompt
        self.tool_fns = tool_fns
        self.input_pydantic_model = input_pydantic_model
        self.state_pydantic_model = state_pydantic_model

    def run(self, input):
        if input is not None:
            assert isinstance(input, self.input_pydantic_model)

        self.state = self.state_pydantic_model()
        self.prompt = self.generate_system_prompt(
            self.input_pydantic_model is not None,
            node_prompt=self.node_prompt,
            node_input=self.input.model_dump_json()
            if self.input_pydantic_model is not None
            else None,
        )

    def generate_system_prompt(self, has_input, kwargs):
        NODE_PROMPT = """
            You are now in the next stage of the conversation. In this stage, the main expectation is the following:
            ```
            {node_prompt}
            ```

            """
        if has_input:
            NODE_PROMPT += """
            There is an input to this stage, which is the output of the previous stage. The input contains
            valuable information that helps you accomplish the main expectation. The input is in JSON format and is the following:
            ```
            {node_input}
            ```

            """

        NODE_PROMPT += """
            During this stage, you must use function calls whenever possible and as soon as possible. If there is
            a user input that has an associated function, you must call it immediately because it will help you with
            accomplishing the user input. When in doubt, use the function/s. In conjunction, you must update a state object whenever possible.
            The state update function is update_state and getting the state function is get_state.
            You cannot proceed to the next stage without updating the state."""

        return NODE_PROMPT.format(**kwargs)

    def update_state(self, state_update):
        self.state = self.state.model_copy(update=state_update)

    def get_state(self):
        return self.state.model_dump_json()


class EdgeSchema:
    def __init__(
        self,
        from_node_schema,
        to_node_schema,
        state_condition_fn,
        new_input_from_state_fn,
    ):
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
        self.new_input_from_state_fn = new_input_from_state_fn

    def check_state_condition(self, state):
        return self.state_condition_fn(state)


## Chain ##


class TakeOrderState(BaseModel):
    order: Order
    has_finished_ordering: bool = Field(
        description="whether the customer has finished ordering. This can only be true after you have explicitly confirmed with them that they have finished ordering, by asking questions like 'Anything else?'."
    )


take_order_node_schema = NodeSchema(
    node_prompt="First, greet the customer. Then, your main job is to take their orders, which also includes answering reasonable questions about the shop & menu only"
    " and assisting them with any issues they may have about their orders. If they are not immediately ready to order after the greeting, you may engage in some small talk"
    " but you need to steer the conversation back to ordering after 4 back-and-forths.",
    tool_fns=[],
    input_pydantic_model=None,
    state_pydantic_model=TakeOrderState,
)


class ConfirmOrderState(BaseModel):
    has_confirmed_order: bool = Field(
        description="whether the customer has confirmed their order"
    )


confirm_order_node_schema = NodeSchema(
    node_prompt="Your main job is to confirm the order with the customer. You do this by repeating the order back to them and get their confirmation.",
    tool_fns=[],
    input_pydantic_model=Order,
    state_pydantic_model=ConfirmOrderState,
)

take_to_confirm_edge_schema = EdgeSchema(
    from_node_schema=take_order_node_schema,
    to_node_schema=confirm_order_node_schema,
    state_condition_fn=lambda state: state.has_finished_ordering,
    new_input_from_state_fn=lambda state: state.order,
)
