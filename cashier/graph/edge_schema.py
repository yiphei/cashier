from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from cashier.graph.base.base_state import BaseStateModel
from cashier.graph.mixin.has_id_mixin import HasIdMixin

if TYPE_CHECKING:
    from cashier.graph.conversation_node import ConversationNode, ConversationNodeSchema

from cashier.graph.base.base_edge_schema import (
    BaseEdgeSchema,
    BaseTransitionConfig,
    BwdStateInit,
    FwdSkipType,
    FwdStateInit,
)
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_status_mixin import Status


class EdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
    def __init__(
        self,
        from_node_schema: ConversationNodeSchema,
        to_node_schema: ConversationNodeSchema,
        transition_config: BaseTransitionConfig,
        new_input_fn: Optional[Callable[[BaseStateModel], Any]] = None,
        bwd_state_init: BwdStateInit = BwdStateInit.RESUME,
        fwd_state_init: FwdStateInit = FwdStateInit.RESET,
        skip_from_complete_to_prev_complete: Optional[
            FwdSkipType
        ] = FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_complete_to_prev_incomplete: Optional[FwdSkipType] = None,
        skip_from_incomplete_to_prev_complete: Optional[
            FwdSkipType
        ] = FwdSkipType.SKIP_IF_INPUT_UNCHANGED,
        skip_from_incomplete_to_prev_incomplete: Optional[FwdSkipType] = None,
    ):
        super().__init__(
            from_node_schema,
            to_node_schema,
            new_input_fn,
            bwd_state_init,
            fwd_state_init,
            skip_from_complete_to_prev_complete,
            skip_from_complete_to_prev_incomplete,
            skip_from_incomplete_to_prev_complete,
            skip_from_incomplete_to_prev_incomplete,
        )
        self.transition_config = transition_config

    def check_transition_config(
        self,
        state: BaseStateModel,
        fn_call,
        is_fn_call_success,
        check_resettable_fields=True,
    ) -> bool:
        return self.transition_config.run_check(
            state,
            fn_call,
            is_fn_call_success,
            check_resettable_fields,
            self.from_node_schema.state_schema.resettable_fields,
        )

    def _check_input(self, state, to_node):
        if self.new_input_fn is not None:
            return self.new_input_fn(state) == to_node.input
        else:
            return to_node.input.__class__(**state) == to_node.input

    def _can_skip(
        self,
        state,
        skip_type: Optional[FwdSkipType],
        to_node: ConversationNode,
    ) -> Tuple[bool, Optional[FwdSkipType]]:
        if skip_type is None:
            return False, skip_type

        if skip_type == FwdSkipType.SKIP:
            return True, skip_type
        elif skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED and self._check_input(
            state, to_node
        ):
            return True, skip_type
        return False, skip_type

    def can_skip(
        self,
        state,
        from_node: ConversationNode,
        to_node: ConversationNode,
        is_prev_from_node_completed: bool,
    ) -> Tuple[bool, Optional[FwdSkipType]]:
        assert from_node.schema == self.from_node_schema
        assert to_node.schema == self.to_node_schema

        if from_node.status == Status.COMPLETED:
            if to_node.status == Status.COMPLETED:
                return self._can_skip(
                    state,
                    self.skip_from_complete_to_prev_complete,
                    to_node,
                )
            else:
                return self._can_skip(
                    state,
                    self.skip_from_complete_to_prev_incomplete,
                    to_node,
                )
        elif is_prev_from_node_completed:
            if to_node.status == Status.COMPLETED:
                return self._can_skip(
                    state,
                    self.skip_from_incomplete_to_prev_complete,
                    to_node,
                )
            else:
                return self._can_skip(
                    state,
                    self.skip_from_incomplete_to_prev_incomplete,
                    to_node,
                )
        else:
            return False, None


class Edge:
    def __init__(self, from_node, to_node, schema):
        self.from_node = from_node
        self.to_node = to_node
        self.schema = schema
