from __future__ import annotations

from abc import ABCMeta
from enum import StrEnum
from typing import Any, List, Literal, Optional, Type, Union, cast, overload

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.base.base_edge_schema import (
    BaseTransitionConfig,
    BwdStateInit,
    FwdStateInit,
)
from cashier.graph.base.base_executable import BaseExecutable
from cashier.graph.base.base_state import BaseStateModel
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.has_status_mixin import HasStatusMixin
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.function_call_context import StateUpdateError
from cashier.tool.tool_registry import ToolRegistry


class Direction(StrEnum):
    FWD = "FWD"
    BWD = "BWD"


class PolyMetaclass(ABCMeta, AutoMixinInit):
    pass


class ConversationNode(
    HasIdMixin, HasStatusMixin, BaseExecutable, metaclass=PolyMetaclass
):
    def __init__(
        self,
        schema: ConversationNodeSchema,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        direction: Direction = Direction.FWD,
    ):
        self.prompt = prompt
        self.input = input
        self.schema = schema
        self.direction = direction
        self.has_run_assistant_turn_before_transition = False
        self.state = state
        self.first_user_message = False
        self.parent = None

    @classmethod
    def init_state(
        cls,
        state_schema: Optional[Type[BaseStateModel]],
        prev_node: Optional[ConversationNode],
        edge_schema: Optional[EdgeSchema],
        direction: Direction,
        input: Any,
    ) -> Optional[BaseStateModel]:
        if state_schema is None:
            return None

        if prev_node is not None:
            state_init_val = getattr(
                edge_schema,
                "fwd_state_init" if direction == Direction.FWD else "bwd_state_init",
            )
            state_init_enum_cls = (
                FwdStateInit if direction == Direction.FWD else BwdStateInit
            )

            if state_init_val == state_init_enum_cls.RESET:  # type: ignore
                return state_schema()
            elif state_init_val == state_init_enum_cls.RESUME or (  # type: ignore
                direction == Direction.FWD
                and state_init_val == state_init_enum_cls.RESUME_IF_INPUT_UNCHANGED  # type: ignore
                and input == prev_node.input
            ):
                return prev_node.state.copy_resume()

        return state_schema()

    def update_state(self, **kwargs: Any) -> None:
        if self.first_user_message:
            old_state = self.state.model_dump()
            new_state = old_state | kwargs
            self.state = self.state.__class__(**new_state)
        else:
            raise StateUpdateError(
                "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
            )

    def get_state(self) -> BaseStateModel:
        return self.state

    def update_first_user_message(self) -> None:
        self.first_user_message = True

    def check_self_completion(self, fn_call, is_fn_call_success):
        self_completion = (
            self.schema.completion_config.run_check(
                self.state, fn_call, is_fn_call_success
            )
            if self.schema.completion_config is not None
            else True
        )
        if self_completion:
            self.mark_as_internally_completed()
        return self_completion

    def check_self_transition(self, fn_call, is_fn_call_success, edge_schemas):
        if self.check_self_completion(fn_call, is_fn_call_success):
            for edge_schema in edge_schemas:
                if edge_schema.check_transition_config(
                    self.state, fn_call, is_fn_call_success
                ):
                    self.mark_as_transitioning()
                    return edge_schema, edge_schema.to_node_schema
        return None, None
    
    def get_next_edge_schema(self):
        return None


class ConversationNodeSchema(HasIdMixin, metaclass=AutoMixinInit):
    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        input_schema: Optional[Type[BaseModel]] = None,
        state_schema: Optional[Type[BaseStateModel]] = None,
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        first_turn: Optional[ModelTurn] = None,
        run_assistant_turn_before_transition: bool = False,
        tool_names: Optional[List[str]] = None,
        completion_config: Optional[BaseTransitionConfig] = None,
    ):
        self.state_schema = state_schema
        self.node_prompt = node_prompt
        self.node_system_prompt = node_system_prompt
        self.input_schema = input_schema
        self.first_turn = first_turn
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition
        self.completion_config = completion_config
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

        if self.state_schema is not None:
            for (
                field_name,
                field_info,
            ) in self.state_schema.model_fields.items():
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
    ) -> ConversationNode: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ConversationNode: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: ConversationNode,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ConversationNode: ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        edge_schema: Optional[EdgeSchema] = None,
        prev_node: Optional[ConversationNode] = None,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ConversationNode:
        state = ConversationNode.init_state(
            self.state_schema, prev_node, edge_schema, direction, input
        )

        prompt = self.node_system_prompt(
            node_prompt=self.node_prompt,
            input=(input.model_dump_json() if self.input_schema is not None else None),
            node_input_json_schema=(
                self.input_schema.model_json_schema() if self.input_schema else None
            ),
            state_json_schema=(
                self.state_schema.model_json_schema() if self.state_schema else None
            ),
            last_msg=last_msg,
            curr_request=curr_request,
        )
        return ConversationNode(
            schema=self,
            input=input,
            state=state,
            prompt=cast(str, prompt),
            direction=direction,
        )
