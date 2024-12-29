import pytest
from deepdiff import DeepDiff
from polyfactory.factories.pydantic_factory import ModelFactory

from cashier.model.model_turn import NodeSystemTurn
from cashier.model.model_util import FunctionCall
from data.graph.airline_change_flight import (
    CHANGE_FLIGHT_GRAPH_SCHEMA,
    get_user_id_node_schema,
)
from data.graph.airline_request import AIRLINE_REQUEST_SCHEMA
from data.types.airline import FlightInfo, ReservationDetails, UserDetails
from tests.base_test import (
    BaseTest,
    TurnArgs,
    assert_number_of_tests,
    get_fn_names_fixture,
)


class TestRequest(BaseTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.start_conv_node_schema = AIRLINE_REQUEST_SCHEMA.start_node_schema
        self.graph_schema = CHANGE_FLIGHT_GRAPH_SCHEMA
        self.edge_schema_id_to_to_cov_node_schema_id = {}
        for (
            node_schema_id,
            edge_schema,
        ) in CHANGE_FLIGHT_GRAPH_SCHEMA.to_conv_node_schema_id_to_edge_schema.items():
            node_schema = (
                CHANGE_FLIGHT_GRAPH_SCHEMA.conv_node_schema_id_to_conv_node_schema[
                    node_schema_id
                ]
            )
            self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id] = node_schema

    def get_edge_schema(self, curr_node_schema):
        return self.graph_schema.from_conv_node_schema_id_to_edge_schema[
            curr_node_schema.id
        ]

    def get_next_conv_node_schema(self, curr_node_schema):
        edge_schema = self.graph_schema.from_conv_node_schema_id_to_edge_schema[
            curr_node_schema.id
        ]
        return self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id]

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
        user_turn = self.add_request_user_turn(agent_executor, "hello", model_provider)

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
        user_turn = self.add_request_user_turn(agent_executor, "hello", model_provider)
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
        user_turn = self.add_request_user_turn(agent_executor, "hello", model_provider)

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
            "i want to change flight",
            model_provider,
            "customer wants to change a flight",
        )
        node_turn = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=get_user_id_node_schema.node_system_prompt(
                    node_prompt=get_user_id_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=get_user_id_node_schema.state_schema.model_json_schema(),
                    last_msg="i want to change flight",
                    curr_request="customer wants to change a flight",
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

    def test_graph_transition(
        self,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
        start_turns,
        is_stream,
    ):
        t1 = self.add_request_user_turn(
            agent_executor,
            "i want to change flight",
            model_provider,
            "customer wants to change a flight",
        )
        node_turn = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=get_user_id_node_schema.node_system_prompt(
                    node_prompt=get_user_id_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=get_user_id_node_schema.state_schema.model_json_schema(),
                    last_msg="i want to change flight",
                    curr_request="customer wants to change a flight",
                ),
                node_id=2,
            ),
        )
        self.build_messages_from_turn(
            node_turn,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

        t2 = self.add_user_turn(
            agent_executor, "my user details are ...", model_provider
        )
        user_details = ModelFactory.create_factory(UserDetails).build()
        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_user_details",
            args={"user_details": user_details.model_dump()},
        )
        t3 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            [fn_call_1],
            {fn_call_1.id: None},
        )

        # --------------------------------

        edge_schema = self.get_edge_schema(
            CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
        )
        next_node_schema = self.get_next_conv_node_schema(
            CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
        )
        input = next_node_schema.get_input(
            agent_executor.graph.curr_node.state, edge_schema
        )

        node_turn_2 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_node_schema.node_system_prompt(
                    node_prompt=next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=next_node_schema.input_schema.model_json_schema(),
                    state_json_schema=next_node_schema.state_schema.model_json_schema(),
                    last_msg="my user details are ...",
                    curr_request="customer wants to change a flight",
                ),
                node_id=3,
            ),
        )
        self.build_messages_from_turn(
            node_turn_2,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

        t4 = self.add_user_turn(
            agent_executor, "my reservation details are ...", model_provider
        )
        res_details = ModelFactory.create_factory(ReservationDetails).build()
        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_reservation_details",
            args={"reservation_details": res_details.model_dump()},
        )
        t5 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            [fn_call_1],
            {fn_call_1.id: None},
        )

        # --------------------------------

        edge_schema = self.get_edge_schema(
            agent_executor.graph.curr_conversation_node.schema
        )
        next_next_node_schema = self.get_next_conv_node_schema(
            agent_executor.graph.curr_conversation_node.schema
        )
        input = next_next_node_schema.get_input(
            agent_executor.graph.curr_node.state, edge_schema
        )

        node_turn_3 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_next_node_schema.node_system_prompt(
                    node_prompt=next_next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=next_next_node_schema.input_schema.model_json_schema(),
                    state_json_schema=next_next_node_schema.state_schema.model_json_schema(),
                    last_msg="my reservation details are ...",
                    curr_request="customer wants to change a flightt",
                ),
                node_id=3,
            ),
        )
        self.build_messages_from_turn(
            node_turn_3,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

        t6 = self.add_user_turn(agent_executor, "the new flight is ...", model_provider)

        flight_info = ModelFactory.create_factory(FlightInfo).build()
        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_flight_infos",
            args={"flight_infos": [flight_info.model_dump()]},
        )
        fn_call_2 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_net_new_cost",
            args={"net_new_cost": 0},
        )
        fn_call_3 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_has_confirmed_new_flights",
            args={"has_confirmed_new_flights": True},
        )
        t7 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            [fn_call_1, fn_call_2, fn_call_3],
            {fn_call_1.id: None, fn_call_2.id: None, fn_call_3.id: None},
        )

        # --------------------------------

        next_next_next_node_schema = self.get_next_conv_node_schema(
            agent_executor.graph.curr_conversation_node.schema
        )
        input_schema, input = (
            agent_executor.graph.curr_node.curr_node.state.get_set_schema_and_fields()
        )
        node_turn_4 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_next_next_node_schema.node_system_prompt(
                    node_prompt=next_next_next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=input_schema.model_json_schema(),
                    state_json_schema=next_next_next_node_schema.state_schema.model_json_schema(),
                    last_msg="the new flight is ...",
                    curr_request="customer wants to change a flightt",
                ),
                node_id=3,
            ),
        )
        self.build_messages_from_turn(
            node_turn_4,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

        t8 = self.add_user_turn(
            agent_executor, "the payment method is ...", model_provider
        )

        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_payment_id",
            args={"payment_id": "123"},
        )
        t9 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            [fn_call_1],
            {fn_call_1.id: None},
        )

        # --------------------------------

        next_next_next_next_node_schema = self.get_next_conv_node_schema(
            agent_executor.graph.curr_conversation_node.schema
        )
        input_schema, input = (
            agent_executor.graph.curr_node.curr_node.state.get_set_schema_and_fields()
        )
        node_turn_5 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_next_next_next_node_schema.node_system_prompt(
                    node_prompt=next_next_next_next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=input_schema.model_json_schema(),
                    state_json_schema=next_next_next_next_node_schema.state_schema.model_json_schema(),
                    last_msg="the payment method is ...",
                    curr_request="customer wants to change a flightt",
                ),
                node_id=3,
            ),
        )
        self.build_messages_from_turn(
            node_turn_5,
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_reservation_flights",
            args={"args": "1"},
        )
        t10 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            [fn_call_1],
            {fn_call_1.id: None},
        )

        TC = self.create_turn_container(
            [
                *start_turns,
                t1,
                node_turn,
                t2,
                t3,
                node_turn_2,
                t4,
                t5,
                node_turn_3,
                t6,
                t7,
                node_turn_4,
                t8,
                t9,
                node_turn_5,
                t10,
            ],
            remove_prev_tool_calls,
        )

        assert not DeepDiff(
            TC.turns[:6], agent_executor.TC.turns[:6], exclude_regex_paths=r".*node_id$"
        )
        # assert len(TC.turns) == len(agent_executor.TC.turns)

        # self.run_assertions(
        #     agent_executor, TC, get_user_id_node_schema.tool_registry, model_provider
        # )


def test_class_test_count(request):
    assert_number_of_tests(TestRequest, __file__, request, 36)
