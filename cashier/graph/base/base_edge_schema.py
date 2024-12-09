from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from cashier.graph.conversation_node import ConversationNodeSchema

from cashier.graph.base.base_state import BaseStateModel


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

    def run_check(
        self,
        state,
        fn_call,
        is_fn_call_success,
        check_resettable_fields=True,
        resettable_fields=None,
    ):
        if isinstance(self, FunctionTransitionConfig):
            return self.check(fn_call, is_fn_call_success)
        elif isinstance(self, StateTransitionConfig):
            return self.check(state, check_resettable_fields, resettable_fields)


class FunctionState(StrEnum):
    CALLED = "CALLED"
    CALLED_AND_SUCCEEDED = "CALLED_AND_SUCCEEDED"


class FunctionTransitionConfig(BaseTransitionConfig):
    fn_name: str
    state: FunctionState

    def check(
        self,
        fn_call,
        is_fn_call_success,
    ):
        if self.state == FunctionState.CALLED:
            return fn_call.name == self.fn_name
        elif self.state == FunctionState.CALLED_AND_SUCCEEDED:
            return fn_call.name == self.fn_name and is_fn_call_success


class StateTransitionConfig(BaseTransitionConfig):
    state_check_fn_map: Dict[str, Callable[[Any], bool]]

    def check(
        self,
        state,
        check_resettable_fields=True,
        resettable_fields=None,
    ):
        for (
            field_name,
            state_check_fn,
        ) in self.state_check_fn_map.items():
            if (
                resettable_fields
                and field_name in resettable_fields
                and not check_resettable_fields
            ):
                continue

            field_value = getattr(state, field_name)
            if not state_check_fn(field_value):
                return False
        return True


class BaseEdgeSchema:
    def __init__(
        self,
        from_node_schema: ConversationNodeSchema,
        to_node_schema: ConversationNodeSchema,
        new_input_fn: Callable[[BaseStateModel], Any],
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
