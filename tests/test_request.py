import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from cashier.model.model_turn import AssistantTurn
from cashier.model.model_util import FunctionCall
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

        t2 = AssistantTurn(
            msg_content=None,
            model_provider=self.fixtures.model_provider,
            tool_registry=self.fixtures.agent_executor.graph.curr_conversation_node.schema.tool_registry,
            fn_calls=[fake_fn_call],
            fn_call_id_to_fn_output={fake_fn_call.id: None},
        )
        self.add_messages_from_turn(t2)

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
            "customer wants to change a flight",
        )
        return [t1, node_turn]

    @pytest.fixture
    def into_second_graph_transition_turns(
        self, agent_executor, into_graph_transition_turns
    ):
        user_details = ModelFactory.create_factory(UserDetails).build()
        fn_call = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_user_details",
            args={"user_details": user_details.model_dump()},
        )

        next_node_schema = self.get_next_conv_node_schema(
            CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
        )

        t_turns_1 = self.add_transition_turns(
            [fn_call],
            {fn_call.id: None},
            "my user details are ...",
            self.get_edge_schema(CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema),
            self.get_next_conv_node_schema(
                CHANGE_FLIGHT_GRAPH_SCHEMA.start_node_schema
            ),
            "customer wants to change a flight",
        )

        t_turns_2 = self.add_new_task(
            ["customer wants to change a flight"],
            [CHANGE_FLIGHT_GRAPH_SCHEMA],
            "change baggage",
            CHANGE_BAGGAGE_GRAPH_SCHEMA,
        )

        res_details = ModelFactory.create_factory(ReservationDetails).build()
        fn_call = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_reservation_details",
            args={"reservation_details": res_details.model_dump()},
        )

        next_next_node_schema = self.get_next_conv_node_schema(next_node_schema)

        t_turns_3 = self.add_transition_turns(
            [fn_call],
            {fn_call.id: None},
            "my reservation details are ...",
            self.get_edge_schema(next_node_schema),
            next_next_node_schema,
            "customer wants to change a flight",
        )

        flight_info = ModelFactory.create_factory(FlightInfo).build()
        fn_call = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_flight_infos",
            args={"flight_infos": [flight_info.model_dump()]},
        )
        fn_call_2 = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_net_new_cost",
            args={"net_new_cost": 0},
        )
        fn_call_3 = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_has_confirmed_new_flights",
            args={"has_confirmed_new_flights": True},
        )

        next_next_next_node_schema = self.get_next_conv_node_schema(
            next_next_node_schema
        )
        t_turns_4 = self.add_transition_turns(
            [fn_call, fn_call_2, fn_call_3],
            {fn_call.id: None, fn_call_2.id: None, fn_call_3.id: None},
            "the new flight is ...",
            self.get_edge_schema(next_next_node_schema),
            next_next_next_node_schema,
            "customer wants to change a flight",
        )

        fn_call = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_state_payment_id",
            args={"payment_id": "123"},
        )
        next_next_next_next_node_schema = self.get_next_conv_node_schema(
            next_next_next_node_schema
        )

        t_turns_5 = self.add_transition_turns(
            [fn_call],
            {fn_call.id: None},
            "the payment method is ...",
            self.get_edge_schema(next_next_next_node_schema),
            next_next_next_next_node_schema,
            "customer wants to change a flight",
        )

        fn_call = FunctionCall.create(
            api_id_model_provider=self.fixtures.model_provider,
            api_id=FunctionCall.generate_fake_id(self.fixtures.model_provider),
            name="update_reservation_flights",
            args={"args": "1"},
        )
        t6 = self.add_assistant_turn(
            None,
            [fn_call],
            {fn_call.id: None},
        )

        fake_fn_call = self.recreate_fake_single_fn_call(
            "think",
            {
                "thought": "I just completed the current request. The next request to be addressed is: change baggage. I must explicitly inform the customer that the current request is completed and that I will address the next request right away. Only after I informed the customer do I receive the tools to address the next request."
            },
        )

        t7 = AssistantTurn(
            msg_content=None,
            model_provider=self.fixtures.model_provider,
            tool_registry=self.fixtures.agent_executor.graph.curr_conversation_node.schema.tool_registry,
            fn_calls=[fake_fn_call],
            fn_call_id_to_fn_output={fake_fn_call.id: None},
        )
        self.add_messages_from_turn(t7)

        t8 = self.add_assistant_turn(
            "finished task",
        )

        # --------------------------------

        t9 = self.add_node_turn(
            luggage_get_user_id_node_schema,
            None,
            "the payment method is ...",
            "change baggage",
        )

        input = luggage_get_reservation_details_node_schema.get_input(
            agent_executor.graph.curr_node.state, edge_1
        )
        t10 = self.add_node_turn(
            luggage_get_reservation_details_node_schema,
            input,
            "the payment method is ...",
            "change baggage",
        )

        # --------------------------------

        new_node_schema = luggage_node_schema
        input = new_node_schema.get_input(agent_executor.graph.curr_node.state, edge_2)

        t11 = self.add_node_turn(
            new_node_schema,
            input,
            "the payment method is ...",
            "change baggage",
        )
        return [
            *t_turns_1,
            *t_turns_2,
            *t_turns_3,
            *t_turns_4,
            *t_turns_5,
            t6,
            t7,
            t8,
            t9,
            t10,
            t11,
        ]

    @pytest.mark.usefixtures("agent_executor")
    def test_graph_initialization(self, start_turns):
        TC = self.create_turn_container(start_turns)
        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_add_user_turn(self, start_turns):
        user_turn = self.add_request_user_turn("hello")

        TC = self.create_turn_container([*start_turns, user_turn])
        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_add_assistant_turn(
        self,
        start_turns,
    ):
        user_turn = self.add_request_user_turn("hello")
        assistant_turn = self.add_assistant_turn("hello back")

        TC = self.create_turn_container([*start_turns, user_turn, assistant_turn])

        self.run_assertions(
            TC,
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

        TC = self.create_turn_container([*start_turns, user_turn, *a_turns])

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.usefixtures("agent_executor")
    def test_node_transition(
        self,
        start_turns,
        into_graph_transition_turns,
    ):
        TC = self.create_turn_container(
            [
                *start_turns,
                *into_graph_transition_turns,
            ],
        )

        self.run_assertions(TC, get_user_id_node_schema.tool_registry)

    @pytest.mark.usefixtures("agent_executor")
    def test_add_new_task(
        self,
        start_turns,
        into_graph_transition_turns,
    ):
        t_turns_1 = self.add_new_task(
            ["customer wants to change a flight"],
            [CHANGE_FLIGHT_GRAPH_SCHEMA],
            "customer wants to change baggage",
            CHANGE_BAGGAGE_GRAPH_SCHEMA,
        )

        TC = self.create_turn_container(
            [
                *start_turns,
                *into_graph_transition_turns,
                *t_turns_1,
            ],
        )
        self.run_assertions(TC, get_user_id_node_schema.tool_registry)

    def test_graph_transition(
        self,
        model_provider,
        agent_executor,
        start_turns,
        into_graph_transition_turns,
        into_second_graph_transition_turns,
    ):
        new_node_schema = luggage_node_schema

        TC = self.create_turn_container(
            [
                *start_turns,
                *into_graph_transition_turns,
                *into_second_graph_transition_turns,
            ],
        )

        self.run_assertions(
            TC,
            new_node_schema.tool_registry,
        )

    def test_default_node(
        self,
        model_provider,
        agent_executor,
        start_turns,
        into_graph_transition_turns,
        into_second_graph_transition_turns,
    ):
        fn_call = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_total_baggages",
            args={"total_baggages": 1},
        )
        fn_call_2 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_nonfree_baggages",
            args={"nonfree_baggages": 1},
        )
        t_turns_12 = self.add_transition_turns(
            [fn_call, fn_call_2],
            {fn_call.id: None, fn_call_2.id: None},
            "the luggage is ...",
            edge_3,
            payment_node_schema,
            "change baggage",
        )

        # fn_call = FunctionCall.create(
        #     api_id_model_provider=model_provider,
        #     api_id=FunctionCall.generate_fake_id(model_provider),
        #     name="update_state_payment_id",
        #     args={"payment_id": "asd"},
        # )
        fn_call = self.create_state_update_fn_call("payment_id", "asd")
        t_turns_13 = self.add_transition_turns(
            [fn_call],
            {fn_call.id: None},
            "the payment method is ...",
            edge_4,
            book_flight_node_schema,
            "change baggage",
        )

        fn_call = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_reservation_baggages",
            args={"args": "1"},
        )
        t14 = self.add_assistant_turn(
            None,
            [fn_call],
            {fn_call.id: None},
        )

        t15 = self.add_node_turn(
            AIRLINE_REQUEST_SCHEMA.default_node_schema,
            None,
            "the payment method is ...",
            None,
        )

        TC = self.create_turn_container(
            [
                *start_turns,
                *into_graph_transition_turns,
                *into_second_graph_transition_turns,
                *t_turns_12,
                *t_turns_13,
                t14,
                t15,
            ],
        )

        self.run_assertions(
            TC,
            AIRLINE_REQUEST_SCHEMA.default_node_schema.tool_registry,
        )


def test_class_test_count(request):
    assert_number_of_tests(TestRequest, __file__, request, 72)
