from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel

from cashier.graph.base.base_executable import BaseExecutableSchema
from cashier.graph.base.base_graph import BaseGraph, BaseGraphSchema
from cashier.graph.conversation_node import ConversationNode, ConversationNodeSchema
from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.mixin.has_id_mixin import HasIdMixin
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

    def handle_skip(
        self,
        fwd_skip_edge_schemas: Set[EdgeSchema],
        TC,
    ) -> Union[Tuple[EdgeSchema, ConversationNodeSchema], Tuple[None, None]]:
        all_node_schemas = {self.curr_node.schema}
        all_node_schemas.update(edge.to_node_schema for edge in fwd_skip_edge_schemas)
        all_node_schemas.update(
            edge.from_node_schema for edge in self.bwd_skip_edge_schemas
        )

        node_schema_id = should_change_node_schema(
            TC, self.curr_node.schema, all_node_schemas, False
        )

        if node_schema_id is not None:
            for edge_schema in fwd_skip_edge_schemas:
                if edge_schema.to_node_schema.id == node_schema_id:
                    return (
                        edge_schema,
                        self.schema.node_schema_id_to_node_schema[node_schema_id],
                    )

            for edge_schema in self.bwd_skip_edge_schemas:
                if edge_schema.from_node_schema.id == node_schema_id:
                    return (
                        edge_schema,
                        self.schema.node_schema_id_to_node_schema[node_schema_id],
                    )

        return None, None

    def handle_wait(
        self,
        fwd_skip_edge_schemas: Set[EdgeSchema],
        TC,
    ) -> Union[Tuple[EdgeSchema, ConversationNodeSchema], Tuple[None, None]]:
        remaining_edge_schemas = (
            set(self.schema.edge_schemas)
            - fwd_skip_edge_schemas
            - self.bwd_skip_edge_schemas
        )

        all_node_schemas = {self.curr_node.schema}
        all_node_schemas.update(edge.to_node_schema for edge in remaining_edge_schemas)

        node_schema_id = should_change_node_schema(
            TC, self.curr_node.schema, all_node_schemas, True
        )

        if node_schema_id is not None:
            return (
                self.to_node_schema_id_to_edge_schema[node_schema_id],
                self.schema.node_schema_id_to_node_schema[node_schema_id],
            )

        return None, None

    def handle_is_off_topic(
        self,
        TC,
    ) -> Union[
        Tuple[EdgeSchema, ConversationNodeSchema, bool], Tuple[None, None, bool]
    ]:
        fwd_skip_edge_schemas = self.compute_fwd_skip_edge_schemas()

        edge_schema, node_schema = self.handle_wait(fwd_skip_edge_schemas, TC)
        if node_schema:
            return edge_schema, node_schema, True  # type: ignore

        edge_schema, node_schema = self.handle_skip(fwd_skip_edge_schemas, TC)
        return edge_schema, node_schema, False  # type: ignore

    def handle_user_turn(self, msg, TC, model_provider, run_off_topic_check=True):
        if not run_off_topic_check or not OffTopicPrompt.run(
            current_node_schema=self.curr_conversation_node.schema,
            tc=TC,
        ):
            edge_schema, node_schema, is_wait = self.curr_graph.handle_is_off_topic(TC)
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
                        edge_schema,
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
        self.compute_bwd_skip_edge_schemas()

    def is_completed(self, fn_call, is_fn_call_success):
        assert self.schema.completion_config is not None
        return (
            self.curr_node.schema == self.schema.last_node_schema
            and self.schema.completion_config.run_check(
                self.state, fn_call, is_fn_call_success
            )
        )
