from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any, Literal, NamedTuple, Optional, overload

from pydantic import BaseModel, ConfigDict

from function_call_context import StateUpdateError
from model_tool_decorator import ToolRegistry
from prompts.node_system import NodeSystemPrompt


class Direction(StrEnum):
    FWD = "FWD"
    BWD = "BWD"


class NodeSchema:
    _counter = 0
    NODE_SCHEMA_ID_TO_NODE_SCHEMA = {}

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
        self.NODE_SCHEMA_ID_TO_NODE_SCHEMA[self.id] = self

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

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: Optional[str] = None,
        prev_node: Literal[None] = None,
        edge_schema: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
    ): ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        prev_node: Node,
        edge_schema: EdgeSchema = None,
        direction: Direction = Direction.FWD,
    ): ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        prev_node: Optional[Node] = None,
        edge_schema: Optional[EdgeSchema] = None,
        direction: Direction = Direction.FWD,
    ):
        state = Node.init_state(
            self.state_pydantic_model, prev_node, edge_schema, direction, input
        )

        prompt = NodeSystemPrompt(
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
            in_edge_schema = prev_node.in_edge_schema
        else:
            in_edge_schema = edge_schema
        return Node(self, input, state, prompt, in_edge_schema, direction)


class Node:
    _counter = 0

    class Status(StrEnum):
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"

    def __init__(
        self, schema, input, state, prompt, in_edge_schema, direction=Direction.FWD
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
    def init_state(cls, state_pydantic_model, prev_node, edge_schema, direction, input):
        if prev_node is not None:
            state_init_val = getattr(
                edge_schema,
                "fwd_state_init" if direction == Direction.FWD else "bwd_state_init",
            )
            state_init_enum_cls = (
                FwdStateInit if direction == Direction.FWD else BwdStateInit
            )

            if state_init_val == state_init_enum_cls.RESET:
                return state_pydantic_model()
            elif state_init_val == state_init_enum_cls.RESUME or (
                direction == Direction.FWD
                and state_init_val == state_init_enum_cls.RESUME_IF_INPUT_UNCHANGED
                and input == prev_node.input
            ):
                return prev_node.state.copy_resume()

        return state_pydantic_model()

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


class BwdStateInit(StrEnum):
    RESET = "RESET"
    RESUME = "RESUME"


class FwdStateInit(StrEnum):
    RESET = "RESET"
    RESUME = "RESUME"
    RESUME_IF_INPUT_UNCHANGED = "RESUME_IF_INPUT_UNCHANGED"


class FwdSkipType(StrEnum):
    SKIP = "SKIP"
    SKIP_IF_INPUT_UNCHANGED = "SKIP_IF_INPUT_UNCHANGED"


class EdgeSchema:
    _counter = 0
    EDGE_SCHEMA_ID_TO_EDGE_SCHEMA = {}

    def __init__(
        self,
        from_node_schema,
        to_node_schema,
        state_condition_fn,
        new_input_from_state_fn,
        bwd_state_init=BwdStateInit.RESUME,
        fwd_state_init=FwdStateInit.RESET,
        skip_from_complete_to_prev_complete=FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_complete_to_prev_incomplete=None,
        skip_from_incomplete_to_prev_complete=FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_incomplete_to_prev_incomplete=None,
    ):
        EdgeSchema._counter += 1
        self.id = EdgeSchema._counter
        self.EDGE_SCHEMA_ID_TO_EDGE_SCHEMA[self.id] = self

        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
        self.new_input_from_state_fn = new_input_from_state_fn
        self.bwd_state_init = bwd_state_init
        self.fwd_state_init = fwd_state_init
        self.skip_from_complete_to_prev_complete = skip_from_complete_to_prev_complete
        self.skip_from_complete_to_prev_incomplete = (
            skip_from_complete_to_prev_incomplete
        )
        # these two below assume that it was previously completed
        self.skip_from_incomplete_to_prev_complete = (
            skip_from_incomplete_to_prev_complete
        )
        self.skip_from_incomplete_to_prev_incomplete = (
            skip_from_incomplete_to_prev_incomplete
        )

    def check_state_condition(self, state):
        return self.state_condition_fn(state)

    def _can_skip(self, skip_type, from_node, to_node):
        if skip_type is None:
            return False, skip_type

        if skip_type == FwdSkipType.SKIP:
            return True, skip_type
        elif (
            skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED
            and self.new_input_from_state_fn(from_node.state) == to_node.input
        ):
            return True, skip_type
        return False, skip_type

    def can_skip(self, from_node, to_node, is_prev_from_node_completed):
        assert from_node.schema == self.from_node_schema
        assert to_node.schema == self.to_node_schema

        if from_node.status == Node.Status.COMPLETED:
            if to_node.status == Node.Status.COMPLETED:
                return self._can_skip(
                    self.skip_from_complete_to_prev_complete,
                    from_node,
                    to_node,
                )
            else:
                return self._can_skip(
                    self.skip_from_complete_to_prev_incomplete,
                    from_node,
                    to_node,
                )
        elif is_prev_from_node_completed:
            if to_node.status == Node.Status.COMPLETED:
                return self._can_skip(
                    self.skip_from_incomplete_to_prev_complete,
                    from_node,
                    to_node,
                )
            else:
                return self._can_skip(
                    self.skip_from_incomplete_to_prev_incomplete,
                    from_node,
                    to_node,
                )


class Edge(NamedTuple):
    from_node: Node
    to_node: Node


class BaseStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def copy_resume(self):
        new_data = copy.deepcopy(dict(self))

        # Iterate through fields and reset those marked as resettable
        for field_name, field_info in self.model_fields.items():
            # Check if field has the resettable marker in its metadata
            if field_info.json_schema_extra and field_info.json_schema_extra.get(
                "resettable"
            ):
                new_data[field_name] = field_info.default

        return self.__class__(**new_data)
