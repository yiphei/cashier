from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from cashier.graph.node_schema import Node, NodeSchema

from cashier.graph.state_model import BaseStateModel


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
        from_node_schema: NodeSchema,
        to_node_schema: NodeSchema,
        state_condition_fn: Callable[[BaseStateModel], bool],
        new_input_fn: Callable[[BaseStateModel, BaseModel], Any],
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

    def check_state_condition(self, state: BaseStateModel) -> bool:
        return self.state_condition_fn(state)

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
