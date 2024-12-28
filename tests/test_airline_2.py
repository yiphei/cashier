import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from cashier.graph.base.base_graph import get_fn_names_fixture
from cashier.model.model_turn import AssistantTurn, NodeSystemTurn
from cashier.model.model_util import FunctionCall
from cashier.tool.function_call_context import StateUpdateError, ToolExceptionWrapper
from data.graph.airline import (
    AIRLINE_REQUEST_SCHEMA,
    BOOK_FLIGHT_GRAPH_SCHEMA,
    find_flight_node_schema,
    get_user_id_node_schema,
)
from data.types.airline import FlightInfo, UserDetails
from tests.base_test import BaseTest, TurnArgs, assert_number_of_tests


class TestRequestAirline(BaseTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.start_conv_node_schema = AIRLINE_REQUEST_SCHEMA.start_node_schema

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
        user_turn = self.add_request_user_turn_2(
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
        user_turn = self.add_request_user_turn_2(
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

    @pytest.mark.parametrize("fn_names", [["inexistent_fn"]])
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
        user_turn = self.add_request_user_turn_2(agent_executor, "hello", model_provider)

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

    # @pytest.mark.parametrize(
    #     "fn_names",
    #     get_fn_names_fixture(get_user_id_node_schema, exclude_update_fn=True),
    # )
    # def test_state_update_before_user_turn(
    #     self,
    #     model_provider,
    #     remove_prev_tool_calls,
    #     is_stream,
    #     fn_names,
    #     agent_executor,
    #     start_turns,
    # ):
    #     fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
    #         model_provider, fn_names, agent_executor.graph.curr_conversation_node
    #     )
    #     fn_call = FunctionCall.create(
    #         name="update_state_user_details",
    #         args={"user_details": None},
    #         api_id_model_provider=model_provider,
    #         api_id=FunctionCall.generate_fake_id(model_provider),
    #     )
    #     fn_calls.append(fn_call)
    #     fn_call_id_to_fn_output[fn_call.id] = ToolExceptionWrapper(
    #         StateUpdateError(
    #             "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
    #         )
    #     )

    #     assistant_turn = self.add_assistant_turn(
    #         agent_executor,
    #         model_provider,
    #         None,
    #         is_stream,
    #         fn_calls,
    #         fn_call_id_to_fn_output,
    #     )

    #     TC = self.create_turn_container(
    #         [*start_turns, assistant_turn], remove_prev_tool_calls
    #     )

    #     self.run_assertions(
    #         agent_executor,
    #         TC,
    #         self.start_conv_node_schema.tool_registry,
    #         model_provider,
    #     )

    # def test_node_transition(
    #     self,
    #     model_provider,
    #     remove_prev_tool_calls,
    #     agent_executor,
    #     start_turns,
    #     first_into_second_transition_turns,
    # ):
    #     TC = self.create_turn_container(
    #         [
    #             *start_turns,
    #             *first_into_second_transition_turns,
    #         ],
    #         remove_prev_tool_calls,
    #     )

    #     self.run_assertions(
    #         agent_executor, TC, find_flight_node_schema.tool_registry, model_provider
    #     )


def test_class_test_count(request):
    assert_number_of_tests(TestRequestAirline, __file__, request, 564)
