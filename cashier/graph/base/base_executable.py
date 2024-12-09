from cashier.graph.mixin.has_status_mixin import Status


class BaseExecutable:
    def check_self_transition(
        self,
        fn_call,
        is_fn_call_success,
        parent_edge_schemas=None,
        new_edge_schema=None,
        new_node_schema=None,
    ):
        if self.check_self_completion(fn_call, is_fn_call_success):
            self.curr_node.mark_as_transitioning()
            self.local_transition_queue.append(self.curr_node)
            self.mark_as_transitioning()
            return None, None
        return new_edge_schema, new_node_schema

    @classmethod
    def get_next_edge_schema(self):
        raise NotImplementedError()

    def check_transition(self, fn_call, is_fn_call_success, parent_edge_schemas=None):
        from cashier.graph.conversation_node import ConversationNode

        if getattr(self, "curr_node", None) is not None:
            if isinstance(self.curr_node, ConversationNode):
                new_edge_schema, new_node_schema = self.curr_node.check_self_transition(
                    fn_call, is_fn_call_success, self.get_next_edge_schema()
                )
            else:
                new_edge_schema, new_node_schema = self.curr_node.check_transition(
                    fn_call, is_fn_call_success, self.get_next_edge_schema()
                )
            if self.curr_node.status == Status.TRANSITIONING:
                self.local_transition_queue.append(self.curr_node)

        return self.check_self_transition(
            fn_call,
            is_fn_call_success,
            parent_edge_schemas,
            new_edge_schema,
            new_node_schema,
        )
