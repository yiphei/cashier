
class NodeSchema:
    NODE_PROMPT = """
    You are now in the next stage of the conversation. In this stage, the main expectation is the following:
    ```
    {node_prompt}
    ```

    There is an input to this stage, which is the output of the previous stage. The input contains
    valuable information that helps you accomplish the main expectation. The input is in JSON format and is the following:
    ```
    {previous_stage_output}
    ```

    During this stage, you must use function calls whenever possible and as soon as possible. If there is
    a user input that has an associated function, you must call it immediately because it will help you with
    accomplishing the user input. When in doubt, use the function/s. In conjunction, you must update a state object whenever possible.
    The state update function is update_state and getting the state function is get_state.
    You cannot proceed to the next stage without updating the state.
    """


    def __init__(self, node_prompt, tool_fns, input_pydantic_model, state_pydantic_model):
        self.prompt = self.NODE_PROMPT.format(node_prompt=node_prompt)
        self.tool_fns = tool_fns
        self.input_pydantic_model = input_pydantic_model
        self.state_pydantic_model = state_pydantic_model

    def run(self, input):
        self.input = input
        self.state = self.state_pydantic_model()
        self.prompt = self.prompt.format(previous_stage_output=input.model_dump_json())

    def update_state(self, state_update):
        self.state = self.state.model_copy(update=state_update)

    def get_state(self):
        return self.state.model_dump_json()

class EdgeSchema:
    def __init__(self, from_node_schema, to_node_schema, state_condition_fn):
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
    
    def check_state_condition(self, state):
        return self.state_condition_fn(state)