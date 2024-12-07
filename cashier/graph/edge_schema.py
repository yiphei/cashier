from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from cashier.graph.mixin.has_id_mixin import HasIdMixin

if TYPE_CHECKING:
    from cashier.graph.conversation_node import ConversationNode

from cashier.graph.base.base_edge_schema import BaseEdgeSchema, FwdSkipType
from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_status_mixin import Status


class EdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
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
        elif (
            skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED
            and self.new_input_fn(state) == to_node.input
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
