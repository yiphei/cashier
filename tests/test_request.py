import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from data.graph.airline_change_baggage import (
    CHANGE_BAGGAGE_GRAPH_SCHEMA,
    book_flight_node_schema,
    edge_1,
    edge_2,
    edge_3,
    edge_4,
)
from data.graph.airline_change_baggage import (
    get_reservation_details_node_schema as luggage_get_reservation_details_node_schema,
)
from data.graph.airline_change_baggage import (
    get_user_id_node_schema as luggage_get_user_id_node_schema,
)
from data.graph.airline_change_baggage import luggage_node_schema, payment_node_schema
from data.graph.airline_change_flight import (
    CHANGE_FLIGHT_GRAPH_SCHEMA,
    get_user_id_node_schema,
)
from data.graph.airline_request import AIRLINE_REQUEST_SCHEMA
from data.types.airline import FlightInfo, ReservationDetails, UserDetails
from tests.base_test import BaseTest, assert_number_of_tests, get_fn_names_fixture


class TestRequest(BaseTest):
    @pytest.fixture(autouse=True)
    def request_schema_input(self):
        return AIRLINE_REQUEST_SCHEMA

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

    def add_new_task(
        self, current_tasks, current_graph_schema_sequence, new_task, new_graph_schema
    ):
        assert self.fixtures.agent_executor.graph.requests == current_tasks
        assert (
            self.fixtures.agent_executor.graph.graph_schema_sequence
            == current_graph_schema_sequence
        )
        t1 = self.add_user_turn(
            "I also want to ...",
            False,
            new_task=new_task,
            task_schema_id=new_graph_schema.id,
        )
        fake_fn_call = self.recreate_fake_single_fn_call(
            "think",
            {
                "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
            },
        )

        t2 = self.add_direct_assistant_turn(
            None,
            [fake_fn_call],
        )

        assert self.fixtures.agent_executor.graph.requests == (
            current_tasks + [new_task]
        )
        assert self.fixtures.agent_executor.graph.graph_schema_sequence == (
            current_graph_schema_sequence + [new_graph_schema]
        )
        return [t1, t2]

    @pytest.fixture
    def start_turns(self, setup_message_dicts, model_provider):
        return [
            self.add_node_turn(
                AIRLINE_REQUEST_SCHEMA.start_node_schema,
                None,
                None,
            )
        ]

    @pytest.fixture
    def into_graph_transition_turns(self, agent_executor, start_turns):
        t1 = self.add_request_user_turn(
            "i want to change flight",
            "customer wants to change a flight",
        )

        node_turn = self.add_node_turn(
            get_user_id_node_schema,
            None,
            "i want to change flight",
        )
        return [*start_turns, t1, node_turn]

    @pytest.fixture
    def into_second_graph_transition_turns(
        self, agent_executor, into_graph_transition_turns
    ):
        t_turns_1 = self.add_chat_turns()
        fn_call = self.create_state_update_fn_call(
            "user_details", pydantic_model=UserDetails
        )

        next_node_schema = self.get_next_conv_node_schema(
            CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
        )

        t_turns_2 = self.add_transition_turns(
            [fn_call],
            "my user details are ...",
            self.get_edge_schema(CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema),
            self.get_next_conv_node_schema(
                CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
            ),
            add_chat_turns=True,
        )

        t_turns_4 = self.add_new_task(
            ["customer wants to change a flight"],
            [CHANGE_FLIGHT_GRAPH_SCHEMA],
            "change baggage",
            CHANGE_BAGGAGE_GRAPH_SCHEMA,
        )

        fn_call = self.create_state_update_fn_call(
            "reservation_details", pydantic_model=ReservationDetails
        )

        next_next_node_schema = self.get_next_conv_node_schema(next_node_schema)

        t_turns_5 = self.add_transition_turns(
            [fn_call],
            "my reservation details are ...",
            self.get_edge_schema(next_node_schema),
            next_next_node_schema,
            add_chat_turns=True,
        )

        flight_info = ModelFactory.create_factory(FlightInfo).build()
        fn_call = self.create_state_update_fn_call(
            "flight_infos", [flight_info.model_dump()]
        )
        special_t = self.add_assistant_turn(None, [fn_call])
        assert (
            self.fixtures.agent_executor.graph.curr_conversation_node.state.flight_infos
            == [flight_info]
        )
        res_details = (
            self.fixtures.agent_executor.graph.curr_conversation_node.input.reservation_details
        )
        old_cost = sum([flight.price for flight in res_details.flights])
        new_cost = flight_info.price
        expected_net_new_cost = new_cost - old_cost
        assert (
            self.fixtures.agent_executor.graph.curr_conversation_node.state.net_new_cost
            == expected_net_new_cost
        )

        fn_call_3 = self.create_state_update_fn_call("has_confirmed_new_flights", True)
        next_next_next_node_schema = self.get_next_conv_node_schema(
            next_next_node_schema
        )
        t_turns_7 = self.add_transition_turns(
            [fn_call_3],
            "the new flight is ...",
            self.get_edge_schema(next_next_node_schema),
            next_next_next_node_schema,
            add_chat_turns=True,
        )
        fn_call = self.create_state_update_fn_call("payment_id", "123")
        next_next_next_next_node_schema = self.get_next_conv_node_schema(
            next_next_next_node_schema
        )

        t_turns_9 = self.add_transition_turns(
            [fn_call],
            "the payment method is ...",
            self.get_edge_schema(next_next_next_node_schema),
            next_next_next_next_node_schema,
        )

        fn_call = self.create_fn_call("update_reservation_flights")
        t10 = self.add_assistant_turn(
            None,
            [fn_call],
        )

        fake_fn_call = self.recreate_fake_single_fn_call(
            "think",
            {
                "thought": "I just completed the current request. The next request to be addressed is: change baggage. I must explicitly inform the customer that the current request is completed and that I will address the next request right away. Only after I informed the customer do I receive the tools to address the next request."
            },
        )

        t11 = self.add_direct_assistant_turn(
            None,
            [fake_fn_call],
        )

        t12 = self.add_assistant_turn(
            "finished task",
        )

        self.curr_request = "change baggage"

        # --------------------------------

        t13 = self.add_node_turn(
            luggage_get_user_id_node_schema,
            None,
            "the payment method is ...",
        )

        input = luggage_get_reservation_details_node_schema.get_input(
            agent_executor.graph.curr_node.state, edge_1
        )
        t14 = self.add_node_turn(
            luggage_get_reservation_details_node_schema,
            input,
            "the payment method is ...",
        )

        # --------------------------------

        new_node_schema = luggage_node_schema
        input = new_node_schema.get_input(agent_executor.graph.curr_node.state, edge_2)

        t15 = self.add_node_turn(
            new_node_schema,
            input,
            "the payment method is ...",
        )
        return [
            *into_graph_transition_turns,
            *t_turns_1,
            *t_turns_2,
            *t_turns_4,
            *t_turns_5,
            special_t,
            *t_turns_7,
            *t_turns_9,
            t10,
            t11,
            t12,
            t13,
            t14,
            t15,
        ]

    @pytest.mark.usefixtures("agent_executor")
    def test_graph_initialization(self, start_turns):
        self.run_assertions(
            start_turns,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_add_user_turn(self, start_turns):
        user_turn = self.add_request_user_turn("hello")
        self.run_assertions(
            [*start_turns, user_turn],
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_add_assistant_turn(
        self,
        start_turns,
    ):
        user_turn = self.add_request_user_turn("hello")
        assistant_turn = self.add_assistant_turn("hello back")

        self.run_assertions(
            [*start_turns, user_turn, assistant_turn],
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.parametrize(
        "fn_names",
        get_fn_names_fixture(
            AIRLINE_REQUEST_SCHEMA.start_node_schema, exclude_all_state_fn=True
        ),
    )
    @pytest.mark.parametrize("separate_fn_calls", [True, False])
    @pytest.mark.usefixtures("agent_executor")
    def test_add_assistant_turn_with_tool_calls(
        self,
        fn_names,
        separate_fn_calls,
        start_turns,
    ):
        user_turn = self.add_request_user_turn("hello")

        if separate_fn_calls:
            tool_names_list = [[fn_name] for fn_name in fn_names]
        else:
            tool_names_list = [fn_names]

        a_turns = []
        for tool_names in tool_names_list:
            assistant_turn = self.add_assistant_turn(None, tool_names=tool_names)
            a_turns.append(assistant_turn)

        self.run_assertions(
            [*start_turns, user_turn, *a_turns],
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_node_transition(
        self,
        into_graph_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        self.run_assertions(
            into_graph_transition_turns + t_turns_1,
            get_user_id_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_add_new_task(
        self,
        into_graph_transition_turns,
    ):
        t_turns_1 = self.add_new_task(
            ["customer wants to change a flight"],
            [CHANGE_FLIGHT_GRAPH_SCHEMA],
            "customer wants to change baggage",
            CHANGE_BAGGAGE_GRAPH_SCHEMA,
        )

        self.run_assertions(
            into_graph_transition_turns + t_turns_1,
            get_user_id_node_schema.tool_registry,
        )

    def test_graph_transition(
        self,
        into_second_graph_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        self.run_assertions(
            into_second_graph_transition_turns + t_turns_1,
            luggage_node_schema.tool_registry,
        )

    def test_skip_node(
        self,
        into_second_graph_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        t_turns_2 = self.add_assistant_turn("hello")
        t_turns_3 = self.add_skip_transition_turns(
            luggage_get_reservation_details_node_schema,
            "hello",
        )

        assert self.fixtures.agent_executor.graph.curr_conversation_node.state.model_dump() == {
            "reservation_details": self.fixtures.agent_executor.graph.curr_node.state.reservation_details.model_dump()
        }

        self.run_assertions(
            into_second_graph_transition_turns + t_turns_1 + [t_turns_2] + t_turns_3,
            luggage_get_reservation_details_node_schema.tool_registry,
        )

    def test_default_node(
        self,
        into_second_graph_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        fn_call = self.create_state_update_fn_call("total_baggages", 1)
        fn_call_2 = self.create_state_update_fn_call("nonfree_baggages", 1)
        t_turns_2 = self.add_transition_turns(
            [fn_call, fn_call_2],
            "the luggage is ...",
            edge_3,
            payment_node_schema,
            add_chat_turns=True,
        )

        fn_call = self.create_state_update_fn_call("payment_id", "asd")
        t_turns_4 = self.add_transition_turns(
            [fn_call],
            "the payment method is ...",
            edge_4,
            book_flight_node_schema,
        )

        fn_call = self.create_fn_call("update_reservation_baggages")
        t5 = self.add_assistant_turn(
            None,
            [fn_call],
        )
        self.curr_request = None

        t6 = self.add_node_turn(
            AIRLINE_REQUEST_SCHEMA.default_node_schema,
            None,
            "the payment method is ...",
        )
        self.run_assertions(
            [
                *into_second_graph_transition_turns,
                *t_turns_1,
                *t_turns_2,
                *t_turns_4,
                t5,
                t6,
            ],
            AIRLINE_REQUEST_SCHEMA.default_node_schema.tool_registry,
        )


def test_class_test_count(request):
    assert_number_of_tests(TestRequest, __file__, request, 80)
