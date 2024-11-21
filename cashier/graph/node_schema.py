from __future__ import annotations

from enum import StrEnum
from typing import Any, List, Literal, Optional, Type, Union, cast, overload

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.edge_schema import BwdStateInit, EdgeSchema, FwdStateInit
from cashier.graph.state_model import BaseStateModel
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.function_call_context import StateUpdateError
from cashier.tool.tool_registry import ToolRegistry


class Direction(StrEnum):
    FWD = "FWD"
    BWD = "BWD"


class Node:
    _counter = 0

    class Status(StrEnum):
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"

    def __init__(
        self,
        schema: NodeSchema,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        in_edge_schema: Optional[EdgeSchema],
        direction: Direction = Direction.FWD,
    ):
        Node._counter += 1
        self.id = Node._counter
        self.state = state
        self.prompt = prompt
        self.input = input
        self.schema = schema
        self.first_user_message = False
        self.status = self.Status.IN_PROGRESS
        self.in_edge_schema = in_edge_schema
        self.direction = direction

    @classmethod
    def init_state(
        cls,
        state_pydantic_model: Type[BaseStateModel],
        prev_node: Optional[Node],
        edge_schema: Optional[EdgeSchema],
        direction: Direction,
        input: Any,
    ) -> BaseStateModel:
        if prev_node is not None:
            state_init_val = getattr(
                edge_schema,
                "fwd_state_init" if direction == Direction.FWD else "bwd_state_init",
            )
            state_init_enum_cls = (
                FwdStateInit if direction == Direction.FWD else BwdStateInit
            )

            if state_init_val == state_init_enum_cls.RESET:  # type: ignore
                return state_pydantic_model()
            elif state_init_val == state_init_enum_cls.RESUME or (  # type: ignore
                direction == Direction.FWD
                and state_init_val == state_init_enum_cls.RESUME_IF_INPUT_UNCHANGED  # type: ignore
                and input == prev_node.input
            ):
                return prev_node.state.copy_resume()

        return state_pydantic_model()

    def mark_as_completed(self) -> None:
        self.status = self.Status.COMPLETED

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


class NodeSchema:
    _counter = 0

    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        input_pydantic_model: Optional[Type[BaseModel]],
        state_pydantic_model: Type[BaseStateModel],
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        first_turn: Optional[ModelTurn] = None,
        tool_names: Optional[List[str]] = None,
    ):
        NodeSchema._counter += 1
        self.id = NodeSchema._counter

        self.node_prompt = node_prompt
        self.node_system_prompt = node_system_prompt
        self.input_pydantic_model = input_pydantic_model
        assert issubclass(state_pydantic_model, BaseStateModel)
        self.state_pydantic_model = state_pydantic_model
        self.first_turn = first_turn
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

        for field_name, field_info in self.state_pydantic_model.model_fields.items():
            new_tool_fn_name = f"update_state_{field_name}"
            field_args = {field_name: (field_info.annotation, field_info)}
            self.tool_registry.add_tool_def(
                new_tool_fn_name,
                f"Function to update the `{field_name}` field in the state",
                field_args,
            )

        self.tool_registry.add_tool_def(
            "get_state", "Function to get the current state, as defined in <state>", {}
        )

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: Literal[None] = None,
        edge_schema: Literal[None] = None,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
    ) -> Node: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
    ) -> Node: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Node,
        direction: Direction = Direction.FWD,
    ) -> Node: ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        edge_schema: Optional[EdgeSchema] = None,
        prev_node: Optional[Node] = None,
        direction: Direction = Direction.FWD,
    ) -> Node:
        state = Node.init_state(
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
            state_json_schema=self.state_pydantic_model.model_json_schema(),
            last_msg=last_msg,
        )

        if direction == Direction.BWD:
            assert prev_node is not None
            in_edge_schema = prev_node.in_edge_schema
        else:
            in_edge_schema = edge_schema
        return Node(self, input, state, cast(str, prompt), in_edge_schema, direction)
