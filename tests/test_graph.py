import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from cashier.graph.request_graph import RequestGraphSchema
from cashier.tool.function_call_context import StateUpdateError, ToolExceptionWrapper
from data.graph.airline_book_flight import BOOK_FLIGHT_GRAPH_SCHEMA
from data.graph.airline_book_flight import (
    get_user_id_node_schema as normal_get_user_id_node_schema,
)
from data.graph.airline_book_flight_and_graph import (
    BOOK_FLIGHT_AND_GRAPH_SCHEMA,
    get_user_id_node_schema,
)
from data.prompt.airline import AirlineNodeSystemPrompt
from data.types.airline import (
    FlightInfo,
    InsuranceValue,
    PassengerInfo,
    PaymentMethod,
    UserDetails,
)
from tests.base_test import BaseTest, assert_number_of_tests, get_fn_names_fixture


@pytest.mark.parametrize(
    "graph_schema, start_conv_node_schema",
    [
        (BOOK_FLIGHT_GRAPH_SCHEMA, normal_get_user_id_node_schema),
        (BOOK_FLIGHT_AND_GRAPH_SCHEMA, get_user_id_node_schema),
    ],
)
class TestGraph(BaseTest):
    @pytest.fixture(autouse=True)
    def request_schema_input(self, graph_schema):
        return RequestGraphSchema(
            node_schemas=[graph_schema],
            edge_schemas=[],
            node_prompt="You are a helpful assistant that helps customers with flight-related requests.",
            node_system_prompt=AirlineNodeSystemPrompt,
            description="Help customers change flights and baggage information for a reservation.",
        )

    @pytest.fixture(autouse=True)
    def setup(self, graph_schema, start_conv_node_schema):
        self.start_conv_node_schema = start_conv_node_schema
        self.graph_schema = graph_schema
        self.edge_schema_id_to_to_cov_node_schema_id = {}
        for (
            node_schema_id,
            edge_schema,
        ) in graph_schema.to_conv_node_schema_id_to_edge_schema.items():
            node_schema = graph_schema.conv_node_schema_id_to_conv_node_schema[
                node_schema_id
            ]
            self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id] = node_schema

        self.ordered_conv_node_schemas = [self.start_conv_node_schema]
        edge_schema = self.get_edge_schema(self.start_conv_node_schema)
        while edge_schema:
            next_node_schema = self.edge_schema_id_to_to_cov_node_schema_id[
                edge_schema.id
            ]
            self.ordered_conv_node_schemas.append(next_node_schema)
            edge_schema = self.get_edge_schema(next_node_schema)

    def get_edge_schema(self, curr_node_schema):
        return self.graph_schema.from_conv_node_schema_id_to_edge_schema.get(
            curr_node_schema.id, None
        )

    def get_next_conv_node_schema(self, curr_node_schema):
        edge_schema = self.graph_schema.from_conv_node_schema_id_to_edge_schema[
            curr_node_schema.id
        ]
        return self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id]

    @pytest.fixture
    def start_turns(self, agent_executor, model_provider, setup_message_dicts):
        second_node_schema = self.start_conv_node_schema
        t1 = self.add_node_turn(
            self.request_schema.start_node_schema,
            None,
            None,
        )
        t2 = self.add_request_user_turn(
            "i want to book flight",
            "customer wants to book flight",
        )
        t3 = self.add_node_turn(
            second_node_schema,
            None,
            "i want to book flight",
        )
        return [t1, t2, t3]

    @pytest.fixture()
    def first_into_second_transition_turns(
        self,
        start_turns,
    ):
        fn_call = self.create_state_update_fn_call(
            "user_details", pydantic_model=UserDetails
        )

        return (
            start_turns
            + self.add_chat_turns()
            + self.add_transition_turns(
                [fn_call],
                "my username is ...",
                self.get_edge_schema(self.start_conv_node_schema),
                self.get_next_conv_node_schema(self.start_conv_node_schema),
            )
        )

    @pytest.fixture()
    def first_into_last_node_turns(
        self,
        first_into_second_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        t2 = self.add_assistant_turn(
            "what flight do you want?",
        )

        flight_info = ModelFactory.create_factory(FlightInfo).build()
        fn_call = self.create_state_update_fn_call(
            "flight_infos", [flight_info.model_dump()]
        )
        t_turns_3 = self.add_transition_turns(
            [fn_call],
            "i want flight from ... to ... on ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[1]),
            self.ordered_conv_node_schemas[2],
            add_chat_turns=True,
        )

        passenger_info = ModelFactory.create_factory(PassengerInfo).build()
        fn_call = self.create_state_update_fn_call(
            "passengers", [passenger_info.model_dump()]
        )
        t_turns_5 = self.add_transition_turns(
            [fn_call],
            "the passengers are ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[2]),
            self.ordered_conv_node_schemas[3],
            add_chat_turns=True,
        )

        fn_call = self.create_state_update_fn_call("add_insurance", InsuranceValue.YES)
        t_turns_7 = self.add_transition_turns(
            [fn_call],
            "the insurance is ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[3]),
            self.ordered_conv_node_schemas[4],
            add_chat_turns=True,
        )

        fn_call = self.create_state_update_fn_call("total_baggages", 1)
        fn_call_2 = self.create_state_update_fn_call("nonfree_baggages", 1)
        t_turns_9 = self.add_transition_turns(
            [fn_call, fn_call_2],
            "the luggage is ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[4]),
            self.ordered_conv_node_schemas[5],
            add_chat_turns=True,
        )

        payment_method = ModelFactory.create_factory(PaymentMethod).build()
        fn_call = self.create_state_update_fn_call(
            "payments", [payment_method.model_dump()]
        )
        fn_call_2 = self.create_state_update_fn_call(
            "has_explained_payment_policy_to_customer", True
        )
        fn_call_3 = self.create_state_update_fn_call("is_payment_finalized", True)
        t_turns_11 = self.add_transition_turns(
            [fn_call, fn_call_2, fn_call_3],
            "the payment is ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[5]),
            self.ordered_conv_node_schemas[6],
        )

        return [
            *first_into_second_transition_turns,
            *t_turns_1,
            t2,
            *t_turns_3,
            *t_turns_5,
            *t_turns_7,
            *t_turns_9,
            *t_turns_11,
        ]

    def test_graph_initialization(self, start_turns):
        self.run_assertions(
            start_turns,
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_user_turn(self, start_turns):
        user_turn = self.add_user_turn("hello")

        self.run_assertions(
            [*start_turns, user_turn],
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_user_turn_with_wait(
        self,
        start_turns,
    ):
        user_turn = self.add_user_turn(
            "hello", False, self.ordered_conv_node_schemas[1].id
        )

        fake_fn_call = self.recreate_fake_single_fn_call(
            "think",
            {
                "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
            },
        )

        assistant_turn = self.add_direct_assistant_turn(
            None,
            [fake_fn_call],
        )

        self.run_assertions(
            [*start_turns, user_turn, assistant_turn],
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_assistant_turn(
        self,
        start_turns,
    ):
        user_turn = self.add_user_turn("hello")
        assistant_turn = self.add_assistant_turn("hello back")

        self.run_assertions(
            [*start_turns, user_turn, assistant_turn],
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.parametrize("fn_names", get_fn_names_fixture(get_user_id_node_schema))
    @pytest.mark.parametrize("separate_fn_calls", [True, False])
    def test_add_assistant_turn_with_tool_calls(
        self,
        fn_names,
        separate_fn_calls,
        start_turns,
    ):
        user_turn = self.add_user_turn("hello")

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

    @pytest.mark.parametrize(
        "fn_names",
        get_fn_names_fixture(get_user_id_node_schema, exclude_update_fn=True) + [[]],
    )
    def test_state_update_before_user_turn(
        self,
        fn_names,
        agent_executor,
        start_turns,
    ):
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            fn_names, agent_executor.graph.curr_conversation_node
        )
        fn_call = self.create_state_update_fn_call(
            "user_details", pydantic_model=UserDetails
        )

        fn_calls.append(fn_call)
        fn_call_id_to_fn_output[fn_call.id] = ToolExceptionWrapper(
            StateUpdateError(
                "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
            )
        )

        assistant_turn = self.add_assistant_turn(
            None,
            fn_calls,
            fn_call_id_to_fn_output,
        )

        self.run_assertions(
            [*start_turns, assistant_turn],
            self.start_conv_node_schema.tool_registry,
        )

    def test_node_transition(
        self,
        first_into_second_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        self.run_assertions(
            first_into_second_transition_turns + t_turns_1,
            self.curr_conversation_node_schema.tool_registry,
        )

    def test_backward_node_skip(
        self,
        first_into_second_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        t2 = self.add_assistant_turn(
            "what flight do you want?",
        )

        t_turns_3 = self.add_skip_transition_turns(
            self.start_conv_node_schema, "what flight do you want?"
        )

        self.run_assertions(
            [*first_into_second_transition_turns, *t_turns_1, t2, *t_turns_3],
            self.start_conv_node_schema.tool_registry,
        )

    def test_forward_node_skip(
        self,
        first_into_second_transition_turns,
    ):
        t_turns_1 = self.add_chat_turns()
        t2 = self.add_assistant_turn(
            "what flight do you want?",
        )

        flight_info = ModelFactory.create_factory(FlightInfo).build()
        fn_call = self.create_state_update_fn_call(
            "flight_infos", [flight_info.model_dump()]
        )
        t_turns_3 = self.add_transition_turns(
            [fn_call],
            "i want flight from ... to ... on ...",
            self.get_edge_schema(self.ordered_conv_node_schemas[1]),
            self.ordered_conv_node_schemas[2],
            add_chat_turns=True,
        )

        t5 = self.add_assistant_turn(
            "thanks for confirming flights, now lets move on to ...",
        )

        t_turns_6 = self.add_skip_transition_turns(
            self.start_conv_node_schema,
            "thanks for confirming flights, now lets move on to ...",
        )

        t_turns_7 = self.add_skip_transition_turns(
            self.ordered_conv_node_schemas[1],
            "what do you want to change?",  # TODO: this is from the default assistant message in add_skip_transition_turns. refactor this
        )

        self.run_assertions(
            [
                *first_into_second_transition_turns,
                *t_turns_1,
                t2,
                *t_turns_3,
                t5,
                *t_turns_6,
                *t_turns_7,
            ],
            self.ordered_conv_node_schemas[1].tool_registry,
        )

    def test_forward_node_skip_special_a(
        self,
        first_into_last_node_turns,
    ):
        t_turns_12 = self.add_skip_transition_turns(
            self.ordered_conv_node_schemas[1],
            "good, lets move on to ...",
            run_wait_node_schema=False,
        )

        t_turn_13 = self.add_assistant_turn(None, tool_names=["get_state"])

        parent_node = self.fixtures.agent_executor.graph.curr_node.conv_node_schema_id_to_parent_node[
            self.ordered_conv_node_schemas[5].id
        ]
        prev_node = parent_node.get_prev_node(self.ordered_conv_node_schemas[5])
        t_14 = self.add_node_turn(
            self.ordered_conv_node_schemas[5],
            prev_node.input,
            "actually, i want to change ...",
        )

        self.run_message_dict_assertions()
        self.run_assertions(
            [
                *first_into_last_node_turns,
                *t_turns_12,
                t_turn_13,
                t_14,
            ],
            self.ordered_conv_node_schemas[5].tool_registry,
        )

    def test_forward_node_skip_special(self, first_into_last_node_turns):

        t_turns_12 = self.add_skip_transition_turns(
            self.ordered_conv_node_schemas[1],
            "good, lets move on to ...",
            run_wait_node_schema=False,
        )
        t_turns_13 = self.add_skip_transition_turns(
            self.ordered_conv_node_schemas[6],
            "what do you want to change?",
        )

        self.run_assertions(
            [
                *first_into_last_node_turns,
                *t_turns_12,
                *t_turns_13,
            ],
            self.ordered_conv_node_schemas[6].tool_registry,
        )


def test_class_test_count(request):
    assert_number_of_tests(TestGraph, __file__, request, 328 * 2)
