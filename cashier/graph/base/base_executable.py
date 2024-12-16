from abc import ABC, abstractmethod

from cashier.graph.mixin.has_status_mixin import Status

class BaseExecutableSchema:
    def __init__(self, state_schema = None, completion_config=None, run_assistant_turn_before_transition=False):
        self.state_schema = state_schema
        self.completion_config = completion_config
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition

class BaseExecutable(ABC):
    def check_self_transition(
        self,
        fn_call,
        is_fn_call_success,
        new_edge_schema=None,
        new_node_schema=None,
    ):
        from cashier.graph.conversation_node import ConversationNode
        from cashier.graph.and_graph_schema import ANDGraph
        if self.check_self_completion(fn_call, is_fn_call_success):
            if not isinstance(self, ANDGraph):
                self.curr_node.mark_as_transitioning()
                self.local_transition_queue.append(self.curr_node)
            self.mark_as_transitioning()
            return None, None
        elif self.curr_node.status == Status.TRANSITIONING and not isinstance(self.curr_node, ConversationNode):
            return self.check_node_transition(fn_call, is_fn_call_success)
        return new_edge_schema, new_node_schema

    def check_node_transition(self, fn_call, is_fn_call_success):
        for edge_schema in self.get_next_edge_schema():
            if edge_schema.check_transition_config(
                self.curr_node.state, fn_call, is_fn_call_success
            ):
                self.curr_node.mark_as_transitioning()
                return edge_schema, edge_schema.to_node_schema

        if not self.get_next_edge_schema():
            self.curr_node.mark_as_transitioning()
            return None, None
        return None, None

    @abstractmethod
    def get_next_edge_schema(self):
        raise NotImplementedError()

    @abstractmethod
    def check_self_completion(self):
        raise NotImplementedError()

    def check_transition(self, fn_call, is_fn_call_success):
        from cashier.graph.conversation_node import ConversationNode

        new_edge_schema, new_node_schema = None, None

        if getattr(self, "curr_node", None) is not None:
            if isinstance(self.curr_node, ConversationNode):
                if self.curr_node.check_self_completion(fn_call, is_fn_call_success):
                    new_edge_schema, new_node_schema = self.check_node_transition(
                        fn_call, is_fn_call_success
                    )
            else:
                new_edge_schema, new_node_schema = self.curr_node.check_transition(
                    fn_call, is_fn_call_success
                )
            if self.curr_node.status == Status.TRANSITIONING:
                self.local_transition_queue.append(self.curr_node)

        return self.check_self_transition(
            fn_call,
            is_fn_call_success,
            new_edge_schema,
            new_node_schema,
        )
