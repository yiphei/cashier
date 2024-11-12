from __future__ import annotations

import copy
from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, Dict, List, Literal, NamedTuple, Optional, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cashier.function_call_context import StateUpdateError
from cashier.tool_registry import ToolRegistry


class Direction(StrEnum):
    FWD = "FWD"
    BWD = "BWD"


class NodeSchema:
    _counter = 0

    def __init__(
        self,
        node_prompt,
        node_system_prompt,
        input_pydantic_model,
        state_pydantic_model,
        tool_registry_or_tool_defs=None,
        first_turn=None,
        tool_names=None,
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
            self.tool_registry = ToolRegistry.create_from_tool_registry(
                tool_registry_or_tool_defs, tool_names
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
                "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
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

    def __init__(
        self,
        from_node_schema,
        to_node_schema,
        state_condition_fn,
        new_input_fn,
        bwd_state_init=BwdStateInit.RESUME,
        fwd_state_init=FwdStateInit.RESET,
        skip_from_complete_to_prev_complete=FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_complete_to_prev_incomplete=None,
        skip_from_incomplete_to_prev_complete=FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_incomplete_to_prev_incomplete=None,
    ):
        EdgeSchema._counter += 1
        self.id = EdgeSchema._counter

        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.state_condition_fn = state_condition_fn
        self.new_input_fn = new_input_fn
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
            and self.new_input_fn(from_node.state, from_node.input) == to_node.input
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


class GraphSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_node_schema: NodeSchema
    edge_schemas: List[EdgeSchema]
    node_schemas: List[NodeSchema]
    node_schema_id_to_node_schema: Optional[Dict[str, NodeSchema]] = None
    edge_schema_id_to_edge_schema: Optional[Dict[str, EdgeSchema]] = None
    from_node_schema_id_to_edge_schema: Optional[Dict[str, List[EdgeSchema]]] = None

    @model_validator(mode="after")
    def init_computed_fields(self):
        self.node_schema_id_to_node_schema = {
            node_schema.id: node_schema for node_schema in self.node_schemas
        }
        self.edge_schema_id_to_edge_schema = {
            edge_schema.id: edge_schema for edge_schema in self.edge_schemas
        }
        self.from_node_schema_id_to_edge_schema = defaultdict(list)
        for edge_schema in self.edge_schemas:
            self.from_node_schema_id_to_edge_schema[
                edge_schema.from_node_schema.id
            ].append(edge_schema)
        return self


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_schema: GraphSchema
    edge_schema_id_to_edges: Dict[str, List[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    from_node_schema_id_to_last_edge_schema_id: Dict[str, str] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    edge_schema_id_to_from_node: Dict[str, None] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )

    def add_fwd_edge(self, from_node, to_node, edge_schema_id):
        self.edge_schema_id_to_edges[edge_schema_id].append(Edge(from_node, to_node))
        self.from_node_schema_id_to_last_edge_schema_id[from_node.schema.id] = (
            edge_schema_id
        )
        self.edge_schema_id_to_from_node[edge_schema_id] = from_node

    def get_edge_by_edge_schema_id(self, edge_schema_id, idx=-1):
        return (
            self.edge_schema_id_to_edges[edge_schema_id][idx]
            if len(self.edge_schema_id_to_edges[edge_schema_id]) >= abs(idx)
            else None
        )

    def get_last_edge_schema_by_from_node_schema_id(self, node_schema_id):
        edge_schema_id = self.from_node_schema_id_to_last_edge_schema_id[node_schema_id]
        return (
            self.graph_schema.edge_schema_id_to_edge_schema[edge_schema_id]
            if edge_schema_id
            else None
        )

    def get_prev_node(self, edge_schema, direction):
        if edge_schema and self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
            from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
            return to_node if direction == Direction.FWD else from_node
        else:
            return None

    def compute_bwd_skip_edge_schemas(self, start_node, curr_bwd_skip_edge_schemas):
        from_node = start_node
        new_edge_schemas = set()
        while from_node.in_edge_schema is not None:
            if from_node.in_edge_schema in curr_bwd_skip_edge_schemas:
                break
            new_edge_schemas.add(from_node.in_edge_schema)
            new_from_node, to_node = self.get_edge_by_edge_schema_id(
                from_node.in_edge_schema.id
            )
            assert from_node == to_node
            from_node = new_from_node

        return new_edge_schemas | curr_bwd_skip_edge_schemas

    def compute_fwd_skip_edge_schemas(self, start_node, start_edge_schemas):
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(start_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
                from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
                if from_node.schema == start_node.schema:
                    from_node = start_node

                if edge_schema.can_skip(
                    from_node,
                    to_node,
                    self.is_prev_from_node_completed(
                        edge_schema, from_node == start_node
                    ),
                )[0]:
                    fwd_jump_edge_schemas.add(edge_schema)
                    next_edge_schema = self.get_last_edge_schema_by_from_node_schema_id(
                        to_node.schema.id
                    )
                    if next_edge_schema:
                        edge_schemas.append(next_edge_schema)

        return fwd_jump_edge_schemas

    def is_prev_from_node_completed(self, edge_schema, is_start_node):
        idx = -1 if is_start_node else -2
        edge = self.get_edge_by_edge_schema_id(edge_schema.id, idx)
        return edge[0].status == Node.Status.COMPLETED if edge else False

    def compute_next_edge_schema(self, start_edge_schema, start_input, curr_node):
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while self.get_edge_by_edge_schema_id(next_edge_schema.id) is not None:
            from_node, to_node = self.get_edge_by_edge_schema_id(next_edge_schema.id)
            if from_node.schema == curr_node.schema:
                from_node = curr_node

            can_skip, skip_type = next_edge_schema.can_skip(
                from_node,
                to_node,
                self.is_prev_from_node_completed(
                    next_edge_schema, from_node == curr_node
                ),
            )

            if can_skip:
                edge_schema = next_edge_schema

                next_next_edge_schema = (
                    self.get_last_edge_schema_by_from_node_schema_id(to_node.schema.id)
                )

                if next_next_edge_schema:
                    next_edge_schema = next_next_edge_schema
                else:
                    input = to_node.input
                    break
            elif skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED:
                if from_node.status != Node.Status.COMPLETED:
                    input = from_node.input
                else:
                    edge_schema = next_edge_schema
                    if from_node != curr_node:
                        input = edge_schema.new_input_fn(
                            from_node.state, from_node.input
                        )
                break
            else:
                if from_node != curr_node:
                    input = from_node.input
                break

        return edge_schema, input

    def add_edge(self, curr_node, new_node, edge_schema, direction=Direction.FWD):
        if direction == Direction.FWD:
            immediate_from_node = curr_node
            if edge_schema.from_node_schema != curr_node.schema:
                from_node = self.edge_schema_id_to_from_node[edge_schema.id]
                immediate_from_node = from_node
                while from_node.schema != curr_node.schema:
                    prev_edge_schema = from_node.in_edge_schema
                    from_node, to_node = self.get_edge_by_edge_schema_id(
                        prev_edge_schema.id
                    )

                self.add_fwd_edge(curr_node, to_node, prev_edge_schema.id)

            self.add_fwd_edge(immediate_from_node, new_node, edge_schema.id)
        elif direction == Direction.BWD:
            if new_node.in_edge_schema:
                from_node, _ = self.get_edge_by_edge_schema_id(
                    new_node.in_edge_schema.id
                )
                self.add_fwd_edge(from_node, new_node, new_node.in_edge_schema.id)

            self.edge_schema_id_to_from_node[edge_schema.id] = new_node
