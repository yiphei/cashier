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



class PromptContainer:
    def __init__(self, base_prompt, prompt_args):
        self.base_prompt = base_prompt
        self.prompt_args = prompt_args

    def format(self, **kwargs):
        return self.base_prompt.format(**kwargs)

class NodeSchema:
    def __init__(self, prompt_container, tool_fns, state_pydantic_model):
        self.prompt_container = prompt_container
        self.tool_fns = tool_fns
        self.state_pydantic_model = state_pydantic_model

    def create_node(self, prompt_args):
        prompt = self.prompt_container.format(prompt_args)
        return Node(prompt, self, self.tool_fns, self.state_pydantic_model)


class Node:
    def __init__(self, prompt, node_schema, tool_fns, state_pydantic_model):
        self.prompt = prompt
        self.node_schema = node_schema
        self.tool_fns = tool_fns
        self.state = state_pydantic_model()
        self.state_json_schema = state_pydantic_model.model_json_schema()

    def update_state(self, state_update):
        self.state = self.state.model_copy(update=state_update)


class EdgeSchema:
    def __init__(self, from_node_schema, to_node_schema, state_condition_fn):
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn