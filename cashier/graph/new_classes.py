from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, List, Literal, Optional, Set, Tuple, Type, Union, cast, overload

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.edge_schema import (
    BwdStateInit,
    Edge,
    EdgeSchema,
    FwdSkipType,
    FwdStateInit,
)
from cashier.graph.node_schema import Direction
from cashier.graph.state_model import BaseStateModel
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.function_call_context import StateUpdateError
from cashier.tool.tool_registry import ToolRegistry


class StateSchemaMixin:
    def __init__(self, state_pydantic_model: Optional[Type[BaseStateModel]]):
        self.state_pydantic_model = state_pydantic_model


class StateMixin:
    def __init__(
        self,
        state: BaseStateModel,
    ):
        self.state = state
        self.first_user_message = False

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


class ActionableSchemaMixin(StateSchemaMixin):
    instance_cls = None

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
        super().__init__(state_pydantic_model)
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
    ) -> ActionableMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ActionableMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: ActionableMixin,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ActionableMixin: ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        edge_schema: Optional[EdgeSchema] = None,
        prev_node: Optional[ActionableMixin] = None,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> ActionableMixin:
        state = ActionableMixin.init_state(
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
        return self.instance_cls(
            self, input, state, cast(str, prompt), in_edge_schema, direction
        )


class ActionableMixin(StateMixin):
    class Status(StrEnum):
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"

    def __init__(
        self,
        schema: ActionableSchemaMixin,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        in_edge_schema: Optional[EdgeSchema],
        direction: Direction = Direction.FWD,
    ):
        super().__init__(state)
        self.prompt = prompt
        self.input = input
        self.schema = schema
        self.status = self.Status.IN_PROGRESS
        self.in_edge_schema = in_edge_schema
        self.direction = direction
        self.has_run_assistant_turn_before_transition = False

    @classmethod
    def init_state(
        cls,
        state_pydantic_model: Optional[Type[BaseStateModel]],
        prev_node: Optional[ActionableMixin],
        edge_schema: Optional[EdgeSchema],
        direction: Direction,
        input: Any,
    ) -> Optional[BaseStateModel]:
        if state_pydantic_model is None:
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


class GraphSchemaMixin:
    def __init__(
        self,
        description: str,
        edge_schemas: List[EdgeSchema],
        node_schemas: List[ActionableMixin],
    ):
        self.description = description
        self.edge_schemas = edge_schemas
        self.node_schemas = node_schemas

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


class GraphMixin:
    def __init__(
        self,
        graph_schema: GraphSchemaMixin,
    ):
        self.graph_schema = graph_schema
        self.edge_schema_id_to_edges = defaultdict(list)
        self.from_node_schema_id_to_last_edge_schema_id = defaultdict(lambda: None)
        self.edge_schema_id_to_from_node = {}

    def add_fwd_edge(
        self, from_node: ActionableMixin, to_node: ActionableMixin, edge_schema_id: int
    ) -> None:
        self.edge_schema_id_to_edges[edge_schema_id].append(Edge(from_node, to_node))
        self.from_node_schema_id_to_last_edge_schema_id[from_node.schema.id] = (
            edge_schema_id
        )
        self.edge_schema_id_to_from_node[edge_schema_id] = from_node

    @overload
    def get_edge_by_edge_schema_id(  # noqa: E704
        self, edge_schema_id: int, idx: int = -1, raise_if_none: Literal[True] = True
    ) -> Edge: ...

    @overload
    def get_edge_by_edge_schema_id(  # noqa: E704
        self, edge_schema_id: int, idx: int = -1, raise_if_none: Literal[False] = False
    ) -> Optional[Edge]: ...

    def get_edge_by_edge_schema_id(
        self, edge_schema_id: int, idx: int = -1, raise_if_none: bool = True
    ) -> Optional[Edge]:
        edge = (
            self.edge_schema_id_to_edges[edge_schema_id][idx]
            if len(self.edge_schema_id_to_edges[edge_schema_id]) >= abs(idx)
            else None
        )
        if edge is None and raise_if_none:
            raise ValueError()
        return edge

    def get_last_edge_schema_by_from_node_schema_id(
        self, node_schema_id: int
    ) -> Optional[EdgeSchema]:
        edge_schema_id = self.from_node_schema_id_to_last_edge_schema_id[node_schema_id]
        return (
            self.graph_schema.edge_schema_id_to_edge_schema[edge_schema_id]
            if edge_schema_id
            else None
        )

    def get_prev_node(
        self, edge_schema: Optional[EdgeSchema], direction: Direction
    ) -> Optional[ActionableMixin]:
        if (
            edge_schema
            and self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
            is not None
        ):
            from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
            return to_node if direction == Direction.FWD else from_node
        else:
            return None

    def compute_bwd_skip_edge_schemas(
        self, start_node: ActionableMixin, curr_bwd_skip_edge_schemas: Set[EdgeSchema]
    ) -> Set[EdgeSchema]:
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

    def compute_fwd_skip_edge_schemas(
        self, start_node: ActionableMixin, start_edge_schemas: Set[EdgeSchema]
    ) -> Set[EdgeSchema]:
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(start_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if (
                self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
                is not None
            ):
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

    def is_prev_from_node_completed(
        self, edge_schema: EdgeSchema, is_start_node: bool
    ) -> bool:
        idx = -1 if is_start_node else -2
        edge = self.get_edge_by_edge_schema_id(edge_schema.id, idx, raise_if_none=False)
        return edge[0].status == ActionableMixin.Status.COMPLETED if edge else False

    def compute_next_edge_schema(
        self,
        start_edge_schema: EdgeSchema,
        start_input: Any,
        curr_node: ActionableMixin,
    ) -> Tuple[EdgeSchema, Any]:
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while (
            self.get_edge_by_edge_schema_id(next_edge_schema.id, raise_if_none=False)
            is not None
        ):
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
                if from_node.status != ActionableMixin.Status.COMPLETED:
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

    def add_edge(
        self,
        curr_node: ActionableMixin,
        new_node: ActionableMixin,
        edge_schema: EdgeSchema,
        direction: Direction = Direction.FWD,
    ) -> None:
        if direction == Direction.FWD:
            immediate_from_node = curr_node
            if edge_schema.from_node_schema != curr_node.schema:
                from_node = self.edge_schema_id_to_from_node[edge_schema.id]
                immediate_from_node = from_node
                while from_node.schema != curr_node.schema:
                    prev_edge_schema = from_node.in_edge_schema
                    from_node, to_node = self.get_edge_by_edge_schema_id(
                        prev_edge_schema.id  # type: ignore
                    )

                self.add_fwd_edge(curr_node, to_node, prev_edge_schema.id)  # type: ignore

            self.add_fwd_edge(immediate_from_node, new_node, edge_schema.id)
        elif direction == Direction.BWD:
            if new_node.in_edge_schema:
                from_node, _ = self.get_edge_by_edge_schema_id(
                    new_node.in_edge_schema.id
                )
                self.add_fwd_edge(from_node, new_node, new_node.in_edge_schema.id)

            self.edge_schema_id_to_from_node[edge_schema.id] = new_node
