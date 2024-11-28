from typing import List, Optional, Type, Union

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.state_model import BaseStateModel
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.tool_registry import ToolRegistry


class StateMixin:
    def __init__(self, state_pydantic_model: Optional[Type[BaseStateModel]]):
        self.state_pydantic_model = state_pydantic_model


class StateWithToolRegistryMixin(StateMixin):

    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        tate_pydantic_model: Optional[Type[BaseStateModel]],
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        tool_names: Optional[List[str]] = None,
        state_pydantic_model: Optional[Type[BaseStateModel]] = None,
        first_turn: Optional[ModelTurn] = None,
        run_assistant_turn_before_transition: bool = False,
        input_pydantic_model: Optional[Type[BaseModel]] = None,
    ):
        super().__init__(state_pydantic_model)
        self.node_prompt = node_prompt
        self.node_system_prompt = node_system_prompt
        self.first_turn = first_turn
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition
        self.input_pydantic_model = input_pydantic_model

        if tool_registry_or_tool_defs is not None and isinstance(
            tool_registry_or_tool_defs, ToolRegistry
        ):
            self.tool_registry = (
                tool_registry_or_tool_defs.__class__.create_from_tool_registry(
                    tool_registry_or_tool_defs, tool_names
                )
            )
        else:
            self.tool_registry = ToolRegistry(tool_registry_or_tool_defs)

        if self.state_pydantic_model is not None:
            for (
                field_name,
                field_info,
            ) in self.state_pydantic_model.model_fields.items():
                new_tool_fn_name = f"update_state_{field_name}"
                field_args = {field_name: (field_info.annotation, field_info)}
                self.tool_registry.add_tool_def(
                    new_tool_fn_name,
                    f"Function to update the `{field_name}` field in the state",
                    field_args,
                )

            self.tool_registry.add_tool_def(
                "get_state",
                "Function to get the current state, as defined in <state>",
                {},
            )
