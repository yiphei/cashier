from __future__ import annotations
from typing import Any, List, Literal, Optional, Type, Union, cast, overload

from pydantic import BaseModel

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_chat_mixin import Direction, HasChatMixin, HasChatSchemaMixin
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.state_mixin import BaseStateModel, HasStateSchemaMixin
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from cashier.tool.tool_registry import ToolRegistry


class Node(HasIdMixin, HasChatMixin, metaclass=AutoMixinInit):
    pass


class NodeSchema(HasIdMixin, HasStateSchemaMixin, metaclass=AutoMixinInit):
    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        input_pydantic_model: Optional[Type[BaseModel]] = None,
        state_pydantic_model: Optional[Type[BaseStateModel]] = None,
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        first_turn: Optional[ModelTurn] = None,
        run_assistant_turn_before_transition: bool = False,
        tool_names: Optional[List[str]] = None,
    ):
        self.state_pydantic_model = state_pydantic_model
        self.node_prompt = node_prompt
        self.node_system_prompt = node_system_prompt
        self.input_pydantic_model = input_pydantic_model
        self.first_turn = first_turn
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition
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

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: Literal[None] = None,
        edge_schema: Literal[None] = None,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasChatMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasChatMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: HasChatMixin,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasChatMixin: ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        edge_schema: Optional[EdgeSchema] = None,
        prev_node: Optional[HasChatMixin] = None,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasChatMixin:
        state = HasChatMixin.init_state(
            self.state_pydantic_model, prev_node, edge_schema, direction, input
        )

        prompt = self.node_system_prompt(
            node_prompt=self.node_prompt,
            input=(
                input.model_dump_json()
                if self.input_pydantic_model is not None
                else None
            ),
            node_input_json_schema=(
                self.input_pydantic_model.model_json_schema()
                if self.input_pydantic_model
                else None
            ),
            state_json_schema=(
                self.state_pydantic_model.model_json_schema()
                if self.state_pydantic_model
                else None
            ),
            last_msg=last_msg,
            curr_request=curr_request,
        )

        if direction == Direction.BWD:
            assert prev_node is not None
            in_edge_schema = prev_node.in_edge_schema
        else:
            in_edge_schema = edge_schema
        return Node(
            schema=self,
            input=input,
            state=state,
            prompt=cast(str, prompt),
            in_edge_schema=in_edge_schema,
            direction=direction,
        )