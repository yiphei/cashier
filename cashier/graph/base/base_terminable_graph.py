from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel

from cashier.graph.base.base_edge_schema import FwdSkipType
from cashier.graph.base.base_executable import BaseExecutableSchema
from cashier.graph.base.base_graph import BaseGraph, BaseGraphSchema
from cashier.graph.conversation_node import (
    ConversationNode,
    ConversationNodeSchema,
    Direction,
)
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.has_id_mixin import HasIdMixin
from cashier.graph.mixin.has_status_mixin import Status
from cashier.model.model_util import FunctionCall, create_think_fn_call
from cashier.prompts.node_schema_selection import NodeSchemaSelectionPrompt
from cashier.prompts.off_topic import OffTopicPrompt
from cashier.turn_container import TurnContainer


def should_change_node_schema(
    TM: TurnContainer,
    current_node_schema: ConversationNodeSchema,
    all_node_schemas: Set[ConversationNodeSchema],
    is_wait: bool,
) -> Optional[int]:
    if len(all_node_schemas) == 1:
        return None
    return NodeSchemaSelectionPrompt.run(
        current_node_schema=current_node_schema,
        tc=TM,
        all_node_schemas=all_node_schemas,
        is_wait=is_wait,
    )


class BaseTerminableGraphSchema(HasIdMixin, BaseGraphSchema, BaseExecutableSchema):
    def __init__(
        self,
        description: str,
        node_schemas: List[ConversationNodeSchema],
        state_schema: Type[BaseModel],
        run_assistant_turn_before_transition: bool = False,
        completion_config=None,
    ):
        HasIdMixin.__init__(self)
        BaseGraphSchema.__init__(self, description, node_schemas)
        BaseExecutableSchema.__init__(
            self,
            state_schema=state_schema,
            completion_config=completion_config,
            run_assistant_turn_before_transition=run_assistant_turn_before_transition,
        )

        self.all_conversation_node_schemas = self.get_leaf_conv_node_schemas()
        self.conversation_node_schema_id_to_conversation_node_schema = {
            node_schema.id: node_schema
            for node_schema in self.all_conversation_node_schemas
        }
        self.to_conversation_node_schema_id_to_edge_schema = {}
        self.from_conversation_node_schema_id_to_edge_schema = {}

        self.from_node_schema_id_to_edge_schema = {
            edge_schema.from_node_schema.id: edge_schema
            for edge_schema in self.get_edge_schemas()
        }

        edge_schemas_stack = self.get_edge_schemas()[:]
        while edge_schemas_stack:
            edge_schema = edge_schemas_stack.pop()
            if isinstance(edge_schema.to_node_schema, BaseGraphSchema):
                schema = (
                    edge_schema.to_node_schema.start_node_schema
                )  # TODO: this and the rest is not truly recursive
                self.to_conversation_node_schema_id_to_edge_schema[schema.id] = (
                    edge_schema
                )
            else:
                self.to_conversation_node_schema_id_to_edge_schema[
                    edge_schema.to_node_schema.id
                ] = edge_schema

            if isinstance(edge_schema.from_node_schema, BaseGraphSchema):
                schema = edge_schema.from_node_schema.last_node_schema
                self.from_conversation_node_schema_id_to_edge_schema[schema.id] = (
                    edge_schema
                )
                edge_schemas_stack.extend(
                    edge_schema.from_node_schema.get_edge_schemas()
                )
            else:
                self.from_conversation_node_schema_id_to_edge_schema[
                    edge_schema.from_node_schema.id
                ] = edge_schema

        # TODO: work in progress
        last_to_node_schema = self.get_edge_schemas()[-1]
        edge_schemas_stack = [last_to_node_schema]
        while edge_schemas_stack:
            edge_schema = edge_schemas_stack.pop()
            if isinstance(edge_schema.to_node_schema, BaseGraphSchema):
                schema = edge_schema.to_node_schema.start_node_schema
                self.to_conversation_node_schema_id_to_edge_schema[schema.id] = (
                    edge_schema
                )
                edge_schemas_stack.extend(
                    edge_schema.to_node_schema.get_edge_schemas()[:].reverse()
                )
            else:
                self.to_conversation_node_schema_id_to_edge_schema[
                    edge_schema.to_node_schema.id
                ] = edge_schema

    def get_edge_schemas(self):
        return self.edge_schemas


class BaseTerminableGraph(BaseGraph):
    def __init__(
        self,
        input: Any,
        request: str,
        schema: BaseGraphSchema,
        edge_schemas=None,
    ):
        input_keys = set(input.keys()) if input is not None else set()
        state = schema.state_schema(**(input or {}))
        state.__pydantic_fields_set__ = input_keys
        super().__init__(input, schema, edge_schemas, request, state=state)
        self.fwd_skip_node_schemas = None

    def _init_skip_node(
        self,
        node_schema,
        direction,
        TC,
    ) -> None:
        from cashier.graph.request_graph import RequestGraph

        prev_node = self.get_prev_node(node_schema)
        assert prev_node is not None

        if isinstance(node_schema, ConversationNodeSchema):
            last_msg = TC.get_asst_message(content_only=True)
            edge_schema = self.get_edge_schema_by_node_schema(node_schema, direction)
            new_node = self.init_node_core(
                node_schema,
                edge_schema,
                prev_node.input,
                last_msg,
                prev_node,
                self.request,
            )
        else:
            edge_schema = self.schema.from_node_schema_id_to_edge_schema[node_schema.id]
            new_node = prev_node

        self.curr_node = new_node
        self.post_node_init(
            edge_schema,
            prev_node,
            TC,
            True,
        )

        if self.parent is not None and not isinstance(self.parent, RequestGraph):
            self.parent._init_skip_node(
                self.schema,
                direction,
                TC,
            )

    def init_skip_node(
        self,
        node_schema: ConversationNodeSchema,
        TC,
    ) -> None:
        direction = Direction.FWD
        if node_schema not in self.fwd_skip_node_schemas:
            direction = Direction.BWD

        parent_node = self.conv_node_schema_id_to_parent_node[node_schema.id]
        parent_node._init_skip_node(
            node_schema,
            direction,
            TC,
        )

    def handle_is_off_topic(
        self,
        TC,
    ) -> Union[
        Tuple[EdgeSchema, ConversationNodeSchema, bool], Tuple[None, None, bool]
    ]:
        self.fwd_skip_node_schemas = self.compute_fwd_skip_node_schemas(True)
        fwd_skip_node_schemas_set = set(self.fwd_skip_node_schemas)

        bwd_skip_node_schemas = self.compute_bwd_skip_node_schemas(True)
        skip_node_schema = fwd_skip_node_schemas_set | bwd_skip_node_schemas
        remaining_node_schemas = (
            set(self.schema.all_conversation_node_schemas) - skip_node_schema
        )
        node_schema_id = should_change_node_schema(
            TC, self.curr_conversation_node.schema, remaining_node_schemas, True
        )
        if node_schema_id is not None:
            return node_schema_id, True  # type: ignore

        all_node_schemas = {self.curr_conversation_node.schema} | skip_node_schema
        node_schema_id = should_change_node_schema(
            TC, self.curr_conversation_node.schema, all_node_schemas, False
        )
        if node_schema_id is not None:
            return self.schema.conversation_node_schema_id_to_conversation_node_schema[node_schema_id], False  # type: ignore
        else:
            return None, False

    def get_bwd_node_schema_and_parent_node(self, node_schema):
        if isinstance(node_schema, BaseGraphSchema):
            return self.get_bwd_node_schema_and_parent_node(
                node_schema.last_node_schema
            )
        return node_schema

    def compute_bwd_skip_node_schemas(self, start_from_curr_node):
        from_node = (
            self.curr_node
            if start_from_curr_node
            else self.get_prev_node(self.schema.last_node_schema)
        )

        if from_node is None:
            return set()

        new_node_schemas = set()
        if isinstance(from_node.schema, BaseGraphSchema):
            new_node_schemas |= from_node.compute_bwd_skip_node_schemas(True)
        while self.to_node_id_to_edge[from_node.id] is not None:
            edge = self.to_node_id_to_edge[from_node.id]

            node_schema = self.get_bwd_node_schema_and_parent_node(
                edge.from_node.schema
            )
            new_node_schemas.add(node_schema)
            assert from_node == edge.to_node
            from_node = edge.from_node
            if isinstance(from_node.schema, BaseGraphSchema):
                new_node_schemas |= from_node.compute_bwd_skip_node_schemas(False)

        return new_node_schemas

    def get_fwd_node_schema_and_parent_node(self, node_schema):
        if isinstance(node_schema, BaseGraphSchema):
            return self.get_fwd_node_schema_and_parent_node(
                node_schema.start_node_schema
            )
        return node_schema

    def compute_fwd_skip_node_schemas(self, start_from_next_edge_schema):
        start_edge_schema = (
            self.next_edge_schema
            if start_from_next_edge_schema
            else self.schema.get_edge_schemas()[0]
        )
        start_node = (
            self.curr_node
            if start_from_next_edge_schema
            else self.get_prev_node(start_edge_schema.from_node_schema)
        )
        if start_node is None or start_edge_schema is None:
            return []

        fwd_jump_node_schemas = []
        edge_schema = start_edge_schema
        next_edge_schema = start_edge_schema
        from_node = start_node

        if isinstance(start_edge_schema.from_node_schema, BaseGraphSchema):
            fwd_jump_node_schemas += start_node.compute_fwd_skip_node_schemas(True)
        while next_edge_schema and (
            self.get_edge_by_edge_schema_id(next_edge_schema.id, raise_if_none=False)
            is not None
        ):
            edge_schema = next_edge_schema
            next_edge_schema = None
            edge = self.get_edge_by_edge_schema_id(edge_schema.id)
            to_node = edge.to_node
            if from_node.schema == start_node.schema:
                from_node = start_node

            if edge_schema.can_skip(
                self.state,
                from_node,
                to_node,
                self.is_prev_from_node_completed(edge_schema, from_node == start_node),
            ):

                node_schema = self.get_fwd_node_schema_and_parent_node(
                    edge_schema.to_node_schema
                )
                fwd_jump_node_schemas.append(node_schema)
                if isinstance(edge_schema.to_node_schema, BaseGraphSchema):
                    graph_node = self.get_prev_node(edge_schema.to_node_schema)
                    fwd_jump_node_schemas += graph_node.compute_fwd_skip_node_schemas(
                        False
                    )
                if self.get_edge_schema_by_from_node_schema_id(to_node.schema.id):
                    next_edge_schema = self.get_edge_schema_by_from_node_schema_id(
                        to_node.schema.id
                    )
                    from_node = to_node

        return fwd_jump_node_schemas

    def compute_next_node_schema(
        self,
        start_node_schema,
        start_input: Any,
    ) -> Tuple[EdgeSchema, Any]:
        self.fwd_skip_node_schemas = self.compute_fwd_skip_node_schemas(True)
        to_node_schema = (
            self.fwd_skip_node_schemas[-1]
            if self.fwd_skip_node_schemas
            else start_node_schema
        )
        to_node = self.get_prev_node(to_node_schema)
        if to_node and to_node.schema == self.curr_node.schema:
            to_node = self.curr_node

        input = to_node.input if self.fwd_skip_node_schemas else start_input

        edge_schema = self.get_edge_schema_by_from_node_schema_id(to_node_schema.id)
        if edge_schema is None:
            return to_node_schema, input

        if (
            self.get_edge_by_edge_schema_id(edge_schema.id, raise_if_none=False)
            is not None
        ):
            from_node = to_node
            edge = self.get_edge_by_edge_schema_id(edge_schema.id)
            to_node = edge.to_node

            if (
                edge_schema.get_skip_type(
                    from_node.status,
                    to_node.status,
                    self.is_prev_from_node_completed(
                        edge_schema, from_node == self.curr_node
                    ),
                )
                == FwdSkipType.SKIP_IF_INPUT_UNCHANGED
                and from_node.status == Status.COMPLETED
            ):
                to_node_schema = edge_schema.to_node_schema
                if from_node != self.curr_node:
                    input = edge_schema.to_node_schema.get_input(
                        from_node.state, edge_schema
                    )

        return to_node_schema, input

    def get_edge_schema_by_to_node_schema(self, node_schema):
        if (
            isinstance(node_schema, ConversationNodeSchema)
            and node_schema.id
            in self.schema.to_conversation_node_schema_id_to_edge_schema
        ):
            return self.schema.to_conversation_node_schema_id_to_edge_schema[
                node_schema.id
            ]
        elif node_schema.id in self.to_node_schema_id_to_edge_schema:
            return self.to_node_schema_id_to_edge_schema.get(node_schema.id, None)

    def pre_init_next_node(
        self,
        node_schema: ConversationNodeSchema,
        input: Any = None,
    ) -> None:
        return self.compute_next_node_schema(node_schema, input)

    def handle_user_turn(self, msg, TC, model_provider, run_off_topic_check=True):
        if not run_off_topic_check or not OffTopicPrompt.run(
            current_node_schema=self.curr_conversation_node.schema,
            tc=TC,
        ):
            node_schema, is_wait = self.handle_is_off_topic(TC)
            if node_schema:
                if is_wait:
                    fake_fn_call = create_think_fn_call(
                        "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
                    )
                    TC.add_assistant_turn(
                        None,
                        model_provider,
                        self.curr_conversation_node.schema.tool_registry,
                        [fake_fn_call],
                        {fake_fn_call.id: None},
                    )
                else:
                    self.init_skip_node(
                        node_schema,
                        TC,
                    )

                    fake_fn_call = FunctionCall.create(
                        api_id=None,
                        api_id_model_provider=None,
                        name="get_state",
                        args={},
                    )
                    TC.add_assistant_turn(
                        None,
                        model_provider,
                        self.curr_conversation_node.schema.tool_registry,
                        [fake_fn_call],
                        {fake_fn_call.id: self.curr_conversation_node.get_state()},
                    )
        self.curr_conversation_node.update_first_user_message()

    def get_next_edge_schema(self):
        return self.from_node_schema_id_to_edge_schema.get(
            self.curr_node.schema.id, None
        )

    def post_node_init(
        self,
        edge_schema: Optional[EdgeSchema],
        prev_node: Optional[ConversationNode],
        TC,
        is_skip: bool = False,
    ) -> None:
        super().post_node_init(
            edge_schema,
            prev_node,
            TC,
            is_skip,
        )
        self.next_edge_schema = self.get_next_edge_schema()

    def is_completed(self, fn_call, is_fn_call_success):
        assert self.schema.completion_config is not None
        return (
            self.curr_node.schema == self.schema.last_node_schema
            and self.schema.completion_config.run_check(
                self.state, fn_call, is_fn_call_success
            )
        )
