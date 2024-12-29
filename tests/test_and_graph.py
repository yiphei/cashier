import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from cashier.model.model_turn import AssistantTurn, NodeSystemTurn
from cashier.model.model_util import FunctionCall
from cashier.tool.function_call_context import StateUpdateError, ToolExceptionWrapper
from data.graph.airline_book_flight import (
    BOOK_FLIGHT_GRAPH_SCHEMA,
    find_flight_node_schema,
    get_user_id_node_schema,
)
from data.graph.airline_request import AIRLINE_REQUEST_SCHEMA
from data.types.airline import FlightInfo, UserDetails
from tests.base_test import (
    BaseTest,
    TurnArgs,
    assert_number_of_tests,
    get_fn_names_fixture,
)


class TestAndGraph(BaseTest):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.start_conv_node_schema = get_user_id_node_schema
        self.graph_schema = BOOK_FLIGHT_GRAPH_SCHEMA
        self.edge_schema_id_to_to_cov_node_schema_id = {}
        for (
            node_schema_id,
            edge_schema,
        ) in BOOK_FLIGHT_GRAPH_SCHEMA.to_conv_node_schema_id_to_edge_schema.items():
            node_schema = (
                BOOK_FLIGHT_GRAPH_SCHEMA.conv_node_schema_id_to_conv_node_schema[
                    node_schema_id
                ]
            )
            self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id] = node_schema

    def get_next_conv_node_schema(self, curr_node_schema):
        edge_schema = self.graph_schema.from_conv_node_schema_id_to_edge_schema[
            curr_node_schema.id
        ]
        return self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id]

    @pytest.fixture
    def start_turns(self, agent_executor, model_provider, setup_message_dicts):
        ut = self.add_request_user_turn(
            "i want to book flight",
            "customer wants to book flight",
        )
        second_node_schema = self.start_conv_node_schema
        turns = [
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
            ut,
            TurnArgs(
                turn=NodeSystemTurn(
                    msg_content=second_node_schema.node_system_prompt(
                        node_prompt=second_node_schema.node_prompt,
                        input=None,
                        node_input_json_schema=None,
                        state_json_schema=second_node_schema.state_schema.model_json_schema(),
                        last_msg="i want to book flight",
                        curr_request="customer wants to book flight",
                    ),
                    node_id=2,
                ),
            ),
        ]

        self.build_messages_from_turn(turns[0])
        self.build_messages_from_turn(turns[2])
        return turns

    @pytest.fixture(params=get_fn_names_fixture(get_user_id_node_schema))
    def first_into_second_transition_turns(
        self,
        agent_executor,
        model_provider,
        is_stream,
        request,
        remove_prev_tool_calls,
    ):
        t1 = self.add_user_turn("hello")
        t2 = self.add_assistant_turn(
            None,
            tool_names=request.param,
        )
        t3 = self.add_user_turn("my username is ...")
        self.run_message_dict_assertions()

        user_details = ModelFactory.create_factory(UserDetails).build()
        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_user_details",
            args={"user_details": user_details.model_dump()},
        )
        t4 = self.add_assistant_turn(
            None,
            [fn_call_1],
            {fn_call_1.id: None},
        )

        next_node_schema = self.get_next_conv_node_schema(self.start_conv_node_schema)

        input_schema, input = (
            agent_executor.graph.curr_node.curr_node.state.get_set_schema_and_fields()
        )
        node_turn = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_node_schema.node_system_prompt(
                    node_prompt=next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=input_schema.model_json_schema(),
                    state_json_schema=next_node_schema.state_schema.model_json_schema(),
                    last_msg="my username is ...",
                    curr_request="customer wants to book flight",
                ),
                node_id=3,
            ),
        )
        self.build_messages_from_turn(
            node_turn,
        )
        return [t1, t2, t3, t4, node_turn]

    def test_graph_initialization(self, start_turns):
        TC = self.create_turn_container(start_turns)
        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_user_turn(self, start_turns):
        user_turn = self.add_user_turn("hello")

        TC = self.create_turn_container([*start_turns, user_turn])
        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_user_turn_with_wait(
        self,
        model_provider,
        start_turns,
    ):
        user_turn = self.add_user_turn("hello", False, find_flight_node_schema.id)

        fake_fn_call = self.recreate_fake_single_fn_call(
            "think",
            {
                "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
            },
        )

        assistant_turn = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_conv_node_schema.tool_registry,
            fn_calls=[fake_fn_call],
            fn_call_id_to_fn_output={fake_fn_call.id: None},
        )
        self.build_messages_from_turn(assistant_turn)

        TC = self.create_turn_container([*start_turns, user_turn, assistant_turn])

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    def test_add_assistant_turn(
        self,
        is_stream,
        start_turns,
    ):
        user_turn = self.add_user_turn("hello")
        assistant_turn = self.add_assistant_turn("hello back")

        TC = self.create_turn_container([*start_turns, user_turn, assistant_turn])

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.parametrize("fn_names", get_fn_names_fixture(get_user_id_node_schema))
    @pytest.mark.parametrize("separate_fn_calls", [True, False])
    def test_add_assistant_turn_with_tool_calls(
        self,
        is_stream,
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

        TC = self.create_turn_container([*start_turns, user_turn, *a_turns])

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    @pytest.mark.parametrize(
        "fn_names",
        get_fn_names_fixture(get_user_id_node_schema, exclude_update_fn=True) + [[]],
    )
    def test_state_update_before_user_turn(
        self,
        model_provider,
        is_stream,
        fn_names,
        agent_executor,
        start_turns,
    ):
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            fn_names, agent_executor.graph.curr_conversation_node
        )
        fn_call = FunctionCall.create(
            name="update_state_user_details",
            args={"user_details": None},
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
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

        TC = self.create_turn_container([*start_turns, assistant_turn])

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    def test_node_transition(
        self,
        start_turns,
        first_into_second_transition_turns,
    ):
        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_transition_turns,
            ],
        )

        self.run_assertions(TC, find_flight_node_schema.tool_registry)

    def test_backward_node_skip(
        self,
        model_provider,
        is_stream,
        agent_executor,
        start_turns,
        first_into_second_transition_turns,
    ):
        t5 = self.add_assistant_turn(
            "what flight do you want?",
        )
        self.run_message_dict_assertions()

        t6 = self.add_user_turn(
            "i want to change my user details",
            False,
            skip_node_schema_id=self.start_conv_node_schema.id,
        )

        node_turn_2 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_conv_node_schema.node_system_prompt(
                    node_prompt=self.start_conv_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=self.start_conv_node_schema.input_from_state_schema,  # just to test that its None
                    state_json_schema=self.start_conv_node_schema.state_schema.model_json_schema(),
                    last_msg="what flight do you want?",
                    curr_request="customer wants to book flight",
                ),
                node_id=4,
            ),
            kwargs={"is_skip": True},
        )
        self.build_messages_from_turn(
            node_turn_2,
            is_skip=True,
        )

        get_state_fn_call = self.recreate_fake_single_fn_call(
            "get_state",
            {},
        )

        t7 = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_conv_node_schema.tool_registry,
            fn_calls=[get_state_fn_call],
            fn_call_id_to_fn_output={
                get_state_fn_call.id: agent_executor.graph.curr_conversation_node.state
            },
        )
        self.build_messages_from_turn(t7)

        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_transition_turns,
                t5,
                t6,
                node_turn_2,
                t7,
            ],
        )

        self.run_assertions(
            TC,
            self.start_conv_node_schema.tool_registry,
        )

    def test_forward_node_skip(
        self,
        model_provider,
        is_stream,
        agent_executor,
        start_turns,
        first_into_second_transition_turns,
    ):
        t5 = self.add_assistant_turn(
            "what flight do you want?",
        )

        t6 = self.add_user_turn(
            "i want flight from ... to ... on ...",
        )
        self.run_message_dict_assertions()

        flight_info = ModelFactory.create_factory(FlightInfo).build()

        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_flight_infos",
            args={"flight_infos": [flight_info.model_dump()]},
        )
        third_fn_calls = [fn_call_1]
        third_fn_calls_fn_call_id_to_fn_output = {
            fn_call.id: None for fn_call in third_fn_calls
        }
        t7 = self.add_assistant_turn(
            None,
            third_fn_calls,
            third_fn_calls_fn_call_id_to_fn_output,
        )

        next_next_node_schema = self.get_next_conv_node_schema(find_flight_node_schema)

        input_schema, input = (
            agent_executor.graph.curr_node.curr_node.state.get_set_schema_and_fields()
        )
        node_turn_2 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_next_node_schema.node_system_prompt(
                    node_prompt=next_next_node_schema.node_prompt,
                    input=input.model_dump_json(),
                    node_input_json_schema=input_schema.model_json_schema(),
                    state_json_schema=next_next_node_schema.state_schema.model_json_schema(),
                    last_msg="i want flight from ... to ... on ...",
                    curr_request="customer wants to book flight",
                ),
                node_id=4,
            ),
        )
        self.build_messages_from_turn(
            node_turn_2,
        )
        t8 = self.add_assistant_turn(
            "thanks for confirming flights, now lets move on to ...",
        )
        self.run_message_dict_assertions()

        t9 = self.add_user_turn(
            "actually, i want to change my user details",
            False,
            skip_node_schema_id=self.start_conv_node_schema.id,
        )
        start_node_schema = self.start_conv_node_schema
        node_turn_3 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=start_node_schema.node_system_prompt(
                    node_prompt=start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=start_node_schema.input_from_state_schema,
                    state_json_schema=start_node_schema.state_schema.model_json_schema(),
                    last_msg="thanks for confirming flights, now lets move on to ...",
                    curr_request="customer wants to book flight",
                ),
                node_id=5,
            ),
            kwargs={"is_skip": True},
        )
        self.build_messages_from_turn(
            node_turn_3,
            is_skip=True,
        )
        get_state_fn_call = self.recreate_fake_single_fn_call("get_state", {})
        t10 = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_conv_node_schema.tool_registry,
            fn_calls=[get_state_fn_call],
            fn_call_id_to_fn_output={
                get_state_fn_call.id: agent_executor.graph.curr_conversation_node.state
            },
        )
        self.build_messages_from_turn(t10)
        t11 = self.add_assistant_turn(
            "what do you want to change?",
        )
        self.run_message_dict_assertions()

        t12 = self.add_user_turn(
            "nvm, nothing",
            False,
            skip_node_schema_id=find_flight_node_schema.id,
        )
        node_turn_4 = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=find_flight_node_schema.node_system_prompt(
                    node_prompt=find_flight_node_schema.node_prompt,
                    input=find_flight_node_schema.input_from_state_schema(
                        **agent_executor.graph.curr_node.curr_node.state.model_dump_fields_set()
                    ).model_dump_json(),
                    node_input_json_schema=find_flight_node_schema.input_from_state_schema.model_json_schema(),
                    state_json_schema=find_flight_node_schema.state_schema.model_json_schema(),
                    last_msg="what do you want to change?",
                    curr_request="customer wants to book flight",
                ),
                node_id=6,
            ),
            kwargs={"is_skip": True},
        )
        self.build_messages_from_turn(
            node_turn_4,
            is_skip=True,
        )
        get_state_fn_call = self.recreate_fake_single_fn_call("get_state", {})
        t13 = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=find_flight_node_schema.tool_registry,
            fn_calls=[get_state_fn_call],
            fn_call_id_to_fn_output={
                get_state_fn_call.id: agent_executor.graph.curr_conversation_node.state
            },
        )
        self.build_messages_from_turn(t13)

        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_transition_turns,
                t5,
                t6,
                t7,
                node_turn_2,
                t8,
                t9,
                node_turn_3,
                t10,
                t11,
                t12,
                node_turn_4,
                t13,
            ],
        )

        self.run_assertions(TC, find_flight_node_schema.tool_registry)


def test_class_test_count(request):
    assert_number_of_tests(TestAndGraph, __file__, request, 564)
