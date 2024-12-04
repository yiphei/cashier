from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from cashier.graph.mixin.has_id_mixin import HasIdMixin

if TYPE_CHECKING:
    from cashier.graph.node_schema import Node

from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.base_edge_schema import BaseEdgeSchema, FwdSkipType


class EdgeSchema(BaseEdgeSchema, HasIdMixin, metaclass=AutoMixinInit):
    def _can_skip(
        self, skip_type: Optional[FwdSkipType], from_node: Node, to_node: Node
    ) -> Tuple[bool, Optional[FwdSkipType]]:
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

    def can_skip(
        self, from_node: Node, to_node: Node, is_prev_from_node_completed: bool
    ) -> Tuple[bool, Optional[FwdSkipType]]:
        from cashier.graph.node_schema import Node

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
        else:
            return False, None


class Edge(NamedTuple):
    from_node: Node
    to_node: Node
