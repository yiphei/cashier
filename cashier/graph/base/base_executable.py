from abc import ABC, abstractmethod
from typing import Any

from cashier.graph.base.base_state import BaseStateModel
from cashier.graph.mixin.has_status_mixin import HasStatusMixin, Status


class BaseExecutableSchema(ABC):
    def __init__(
        self,
        state_schema=None,
        completion_config=None,
        run_assistant_turn_before_transition=False,
    ):
        self.state_schema = state_schema
        self.completion_config = completion_config
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition

    @abstractmethod
    def get_input(self, state, edge_schema):
        raise NotImplementedError()

    @abstractmethod
    def create_node(self, input, last_msg, edge_schema, prev_node, direction, request):
        raise NotImplementedError()


class BaseExecutable(ABC, HasStatusMixin):
    def __init__(self, state):
        self.state = state
        HasStatusMixin.__init__(self)

    @abstractmethod
    def is_completed(self):
        raise NotImplementedError()

    def update_state(self, **kwargs: Any) -> None:
        old_state = self.state.model_dump()
        old_state_fields_set = self.state.model_fields_set
        new_state = old_state | kwargs
        new_state_fields_set = old_state_fields_set | kwargs.keys()
        self.state = self.state.__class__(**new_state)
        self.state.__pydantic_fields_set__ = new_state_fields_set

    def update_state_from_executable(self, executable):
        state = executable.state
        self.update_state(**state.model_dump(exclude=state.resettable_fields))

    def get_state(self) -> BaseStateModel:
        return self.state
    
    def check_transition(self, fn_call, is_fn_call_success):
        if self.is_completed(fn_call, is_fn_call_success):
            self.mark_as_internally_completed()
        return None, None


class BaseGraphExecutable(BaseExecutable):
    def check_node_transition(self, fn_call, is_fn_call_success):
        assert self.curr_node.status == Status.INTERNALLY_COMPLETED
        for edge_schema in self.get_next_edge_schema():
            if edge_schema.check_transition_config(
                self.curr_node.state, fn_call, is_fn_call_success
            ):
                self.curr_node.mark_as_transitioning()
                self.local_transition_queue.append(self.curr_node)
                return edge_schema, edge_schema.to_node_schema

        return None, None

    @abstractmethod
    def get_next_edge_schema(self):
        raise NotImplementedError()

    def check_transition(self, fn_call, is_fn_call_success):
        new_edge_schema, new_node_schema = None, None

        if getattr(self, "curr_node", None) is not None:
            new_edge_schema, new_node_schema = self.curr_node.check_transition(
                fn_call, is_fn_call_success
            )

        if self.is_completed(fn_call, is_fn_call_success):
            if self.curr_node.status == Status.INTERNALLY_COMPLETED:
                self.curr_node.mark_as_transitioning()
                self.local_transition_queue.append(self.curr_node)
            self.mark_as_internally_completed()
            return None, None
        elif self.curr_node.status == Status.INTERNALLY_COMPLETED:
            return self.check_node_transition(fn_call, is_fn_call_success)
        return new_edge_schema, new_node_schema
