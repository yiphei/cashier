from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from cashier.graph.node_schema import Node, NodeSchema

from cashier.graph.auto_mixin_init import AutoMixinInit
from cashier.graph.has_id_mixin import HasIdMixin
from cashier.graph.state import BaseStateModel


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


class BaseTransitionConfig(BaseModel):
    need_user_msg: bool


class FunctionState(StrEnum):
    CALLED = "CALLED"
    CALLED_AND_SUCCEEDED = "CALLED_AND_SUCCEEDED"


class FunctionTransitionConfig(BaseTransitionConfig):
    fn_name: str
    state: FunctionState


class StateTransitionConfig(BaseTransitionConfig):
    state_check_fn: Callable[[BaseStateModel], bool]


class BaseEdgeSchema:
    def __init__(
        self,
        from_node_schema: NodeSchema,
        to_node_schema: NodeSchema,
        transition_config: BaseTransitionConfig,
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
        self.from_node_schema = from_node_schema
        self.to_node_schema = to_node_schema
        self.transition_config = transition_config
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

    def check_transition_config(
        self, state: BaseStateModel, fn_call, is_fn_call_success
    ) -> bool:
        if isinstance(self.transition_config, FunctionTransitionConfig):
            if self.transition_config.state == FunctionState.CALLED:
                return fn_call.name == self.transition_config.fn_name
            elif self.transition_config.state == FunctionState.CALLED_AND_SUCCEEDED:
                return (
                    fn_call.name == self.transition_config.fn_name
                    and is_fn_call_success
                )
        elif isinstance(self.transition_config, StateTransitionConfig):
            return self.transition_config.state_check_fn(state)


class Edge(NamedTuple):
    from_node: Node
    to_node: Node
