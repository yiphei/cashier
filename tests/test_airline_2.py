import pytest

from cashier.model.model_turn import NodeSystemTurn
from data.graph.airline import (
    AIRLINE_REQUEST_SCHEMA,
    BOOK_FLIGHT_GRAPH_SCHEMA,
    get_user_id_node_schema,
)
from tests.base_test import (
    BaseTest,
    TurnArgs,
    assert_number_of_tests,
    get_fn_names_fixture,
)


class TestRequestAirline(BaseTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.start_conv_node_schema = AIRLINE_REQUEST_SCHEMA.start_node_schema
        self.graph_schema = BOOK_FLIGHT_GRAPH_SCHEMA

    @pytest.fixture(autouse=True)
    def setup_start_message_list(
        self, start_turns, setup_message_dicts, model_provider
    ):
        self.build_messages_from_turn(start_turns[0], model_provider)

    @pytest.fixture
    def start_turns(self):
        return [
            TurnArgs(
                turn=NodeSystemTurn(
                    msg_content=AIRLINE_REQUEST_SCHEMA.start_node_schema.node_system_prompt(
                        node_prompt=AIRLINE_REQUEST_SCHEMA.start_node_schema.node_prompt,
                        input=None,
                        node_input_json_schema=None,
                        state_json_schema=None,
                        last_msg=None,
                        curr_request=None,
                    ),
                    node_id=1,
                ),
            ),
        ]

    def test_graph_initialization(
        self, model_provider, remove_prev_tool_calls, agent_executor, start_turns
    ):
        TC = self.create_turn_container(start_turns, remove_prev_tool_calls)
        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_user_turn(
        self, model_provider, remove_prev_tool_calls, agent_executor, start_turns
    ):
        user_turn = self.add_request_user_turn(
            agent_executor, "hello", model_provider
        )

        TC = self.create_turn_container(
            [*start_turns, user_turn], remove_prev_tool_calls
        )
        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_assistant_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
        start_turns,
    ):
        user_turn = self.add_request_user_turn(
            agent_executor, "hello", model_provider
        )
        assistant_turn = self.add_assistant_turn(
            agent_executor, model_provider, "hello back", is_stream
        )

        TC = self.create_turn_container(
            [*start_turns, user_turn, assistant_turn], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    @pytest.mark.parametrize(
        "fn_names",
        get_fn_names_fixture(
            AIRLINE_REQUEST_SCHEMA.start_node_schema, exclude_all_state_fn=True
        ),
    )
    @pytest.mark.parametrize("separate_fn_calls", [True, False])
    def test_add_assistant_turn_with_tool_calls(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        fn_names,
        separate_fn_calls,
        agent_executor,
        start_turns,
    ):
        user_turn = self.add_request_user_turn(
            agent_executor, "hello", model_provider
        )

        if separate_fn_calls:
            tool_names_list = [[fn_name] for fn_name in fn_names]
        else:
            tool_names_list = [fn_names]

        a_turns = []
        for tool_names in tool_names_list:
            assistant_turn = self.add_assistant_turn(
                agent_executor, model_provider, None, is_stream, tool_names=tool_names
            )
            a_turns.append(assistant_turn)

        TC = self.create_turn_container(
            [*start_turns, user_turn, *a_turns], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_node_transition(
        self,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
        start_turns,
    ):
        t1 = self.add_request_user_turn(
            agent_executor,
            "i want to book flight",
            model_provider,
            "customer wants to book a flight",
        )
        graph_schema_start_node = get_user_id_node_schema
        node_turn = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=graph_schema_start_node.node_system_prompt(
                    node_prompt=graph_schema_start_node.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=graph_schema_start_node.state_schema.model_json_schema(),
                    last_msg="i want to book flight",
                    curr_request="customer wants to book a flight",
                ),
                node_id=2,
            ),
        )
        self.build_messages_from_turn(
            node_turn,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )
        TC = self.create_turn_container(
            [
                *start_turns,
                t1,
                node_turn,
            ],
            remove_prev_tool_calls,
        )

        self.run_assertions(
            agent_executor, TC, get_user_id_node_schema.tool_registry, model_provider
        )


def test_class_test_count(request):
    assert_number_of_tests(TestRequestAirline, __file__, request, 36)
