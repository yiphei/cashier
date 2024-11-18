from collections import defaultdict
from contextlib import ExitStack
from io import StringIO
from typing import Any, Dict
from unittest.mock import Mock, call, patch

import pytest
from deepdiff import DeepDiff
from pydantic import BaseModel, Field

from cashier.agent_executor import AgentExecutor
from cashier.graph import Node
from cashier.model.model_client import AnthropicModelOutput, Model, OAIModelOutput
from cashier.model.model_turn import AssistantTurn, ModelTurn, NodeSystemTurn, UserTurn
from cashier.model.model_util import FunctionCall, ModelProvider
from cashier.tool.function_call_context import (
    InexistentFunctionError,
    StateUpdateError,
    ToolExceptionWrapper,
)
from cashier.turn_container import TurnContainer
from data.graph.cashier import cashier_graph_schema
from data.tool_registry.cashier_tool_registry import CupSize, ItemOrder, Order


class TurnArgs(BaseModel):
    turn: ModelTurn
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = Mock(spec=Model)
        self.stdout_patcher = patch("sys.stdout", new_callable=StringIO)
        self.stdout_patcher.start()
        Node._counter = 0
        self.start_node_schema = cashier_graph_schema.start_node_schema

        yield

        self.stdout_patcher.stop()
        self.model.reset_mock()

    def create_turn_container(self, turn_args_list):
        TC = TurnContainer()
        for turn_args in turn_args_list:
            add_fn = None
            if isinstance(turn_args, TurnArgs):
                turn = turn_args.turn
                kwargs = {"turn": turn_args.turn, **turn_args.kwargs}
            else:
                turn = turn_args
                kwargs = {"turn": turn_args}

            if isinstance(turn, NodeSystemTurn):
                add_fn = "add_node_turn"
            elif isinstance(turn, AssistantTurn):
                add_fn = "add_assistant_turn"
            elif isinstance(turn, UserTurn):
                add_fn = "add_user_turn"

            for mm in TC.model_provider_to_message_manager.values():
                getattr(mm, add_fn)(**kwargs)

            TC.turns.append(turn)
        return TC

    def create_mock_model_completion(
        self,
        model_provider,
        message=None,
        is_stream=False,
        message_prop=None,
        prob=None,
        fn_calls=None,
    ):
        model_completion_class = (
            OAIModelOutput
            if model_provider == ModelProvider.OPENAI
            else AnthropicModelOutput
        )
        fn_calls = fn_calls or []

        model_completion = model_completion_class(output_obj=None, is_stream=is_stream)
        model_completion.msg_content = message
        model_completion.get_message = Mock(return_value=message)
        if message is not None:
            model_completion.stream_message = Mock(
                return_value=iter(message.split(" "))
            )
        else:
            model_completion.stream_message = Mock(return_value=None)
        model_completion.get_fn_calls = Mock(return_value=iter(fn_calls))
        model_completion.stream_fn_calls = Mock(return_value=iter(fn_calls))
        model_completion.fn_calls = fn_calls
        if message_prop is not None:
            model_completion.get_message_prop = Mock(return_value=message_prop)
            if model_provider == ModelProvider.OPENAI:
                model_completion.get_prob = Mock(return_value=prob)
        return model_completion

    def create_fake_fn_calls(self, model_provider, fn_names, node):
        fn_calls = []
        tool_registry = node.schema.tool_registry
        fn_call_id_to_fn_output = {}
        for fn_name in fn_names:
            args = {"arg_1": "arg_1_val"}
            if fn_name == "get_state":
                args = {}
            elif fn_name.startswith("update_state"):
                field_name = fn_name.removeprefix("update_state_")
                model_fields = node.schema.state_pydantic_model.model_fields
                field_info = model_fields[field_name]

                # Get default value or call default_factory if it exists
                default_value = (
                    field_info.default_factory()
                    if field_info.default_factory is not None
                    else field_info.default
                )
                args = {field_name: default_value}

            fn_call = FunctionCall.create_fake_fn_call(
                model_provider,
                fn_name,
                args=args,
            )
            fn_calls.append(fn_call)
            if fn_name in tool_registry.tool_names:
                if fn_name.startswith("update_state"):
                    output = None
                elif fn_name == "get_state":
                    output = node.state
                else:
                    output = f"{fn_name}'s output"

                fn_call_id_to_fn_output[fn_call.id] = output
            else:
                fn_call_id_to_fn_output[fn_call.id] = ToolExceptionWrapper(
                    InexistentFunctionError(fn_name)
                )

        return fn_calls, fn_call_id_to_fn_output

    def add_user_turn(
        self,
        agent_executor,
        message,
        model_provider,
        is_on_topic,
        fwd_skip_node_schema_id=None,
        bwd_skip_node_schema_id=None,
    ):
        model_chat_side_effects = []

        is_on_topic_model_completion = self.create_mock_model_completion(
            model_provider, None, False, is_on_topic, 0.5
        )
        model_chat_side_effects.append(is_on_topic_model_completion)

        if not is_on_topic:
            is_wait_model_completion = self.create_mock_model_completion(
                model_provider,
                None,
                False,
                fwd_skip_node_schema_id or agent_executor.curr_node.schema.id,
                0.5,
            )
            model_chat_side_effects.append(is_wait_model_completion)

            if fwd_skip_node_schema_id is None:
                bwd_skip_model_completion = self.create_mock_model_completion(
                    model_provider,
                    None,
                    False,
                    bwd_skip_node_schema_id or agent_executor.curr_node.schema.id,
                    0.5,
                )
                model_chat_side_effects.append(bwd_skip_model_completion)

        self.model.chat.side_effect = model_chat_side_effects
        agent_executor.add_user_turn(message)

    def add_assistant_turn(
        self,
        agent_executor,
        model_provider,
        message,
        is_stream,
        fn_calls=None,
        fn_call_id_to_fn_output=None,
    ):
        model_completion = self.create_mock_model_completion(
            model_provider, message, is_stream, fn_calls=fn_calls
        )
        get_state_fn_call = (
            next((fn_call for fn_call in fn_calls if fn_call.name == "get_state"), None)
            if fn_calls
            else None
        )
        update_state_fn_calls = (
            [fn_call for fn_call in fn_calls if fn_call.name.startswith("update_state")]
            if fn_calls
            else []
        )

        tool_registry = agent_executor.curr_node.schema.tool_registry

        fn_calls = fn_calls or []
        expected_calls_map = defaultdict(list)
        for fn_call in fn_calls:
            expected_calls_map[fn_call.name].append(call(**fn_call.args))

        with patch.dict(
            tool_registry.fn_name_to_fn,
            {
                fn_call.name: Mock(return_value=fn_call_id_to_fn_output[fn_call.id])
                for fn_call in fn_calls
            },
        ) as patched_fn_name_to_fn, ExitStack() as stack:
            curr_node = agent_executor.curr_node
            if get_state_fn_call is not None:
                stack.enter_context(
                    patch.object(
                        agent_executor.curr_node,
                        "get_state",
                        wraps=agent_executor.curr_node.get_state,
                    )
                )

            if update_state_fn_calls:
                stack.enter_context(
                    patch.object(
                        agent_executor.curr_node,
                        "update_state",
                        wraps=agent_executor.curr_node.update_state,
                    )
                )

            agent_executor.add_assistant_turn(model_completion)

            visited_fn_call_ids = set()
            for fn_call in fn_calls:
                if fn_call.id in visited_fn_call_ids:
                    continue
                visited_fn_call_ids.add(fn_call.id)

                patched_fn = None
                if fn_call == get_state_fn_call:
                    patched_fn = curr_node.get_state
                elif fn_call in update_state_fn_calls:
                    patched_fn = curr_node.update_state
                else:
                    patched_fn = patched_fn_name_to_fn[fn_call.name]

                if fn_call.name not in tool_registry.tool_names:
                    patched_fn.assert_not_called()
                else:
                    patched_fn.assert_has_calls(expected_calls_map[fn_call.name])

    @pytest.fixture
    def agent_executor(self, model_provider, remove_prev_tool_calls):
        return AgentExecutor(
            model=self.model,
            elevenlabs_client=None,
            graph_schema=cashier_graph_schema,
            audio_output=False,
            remove_prev_tool_calls=remove_prev_tool_calls,
            model_provider=model_provider,
        )

    @classmethod
    @pytest.fixture(params=[ModelProvider.OPENAI, ModelProvider.ANTHROPIC])
    def model_provider(cls, request):
        return request.param

    @classmethod
    @pytest.fixture(params=[True, False])
    def remove_prev_tool_calls(cls, request):
        return request.param

    @classmethod
    @pytest.fixture(params=[True, False])
    def is_stream(cls, request):
        return request.param

    @classmethod
    @pytest.fixture(
        params=[
            ["get_menu_item_from_name"],
            ["get_state"],
            ["update_state_order"],
            ["inexistent_fn"],
            ["get_menu_item_from_name", "get_menu_item_from_name"],
            ["get_state", "update_state_order"],
            ["get_state", "update_state_order", "inexistent_fn"],
            ["get_state", "get_menu_item_from_name", "update_state_order"],
            [
                "get_state",
                "get_menu_item_from_name",
                "update_state_order",
                "get_menu_item_from_name",
            ],
        ]
    )
    def fn_names(cls, request):
        return request.param

    def test_initial_node(self, remove_prev_tool_calls, agent_executor):
        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )

        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn

        TC = self.create_turn_container([FIRST_TURN, SECOND_TURN])
        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def test_add_user_turn(
        self, model_provider, remove_prev_tool_calls, agent_executor
    ):
        self.add_user_turn(agent_executor, "hello", model_provider, True)

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")

        TC = self.create_turn_container([FIRST_TURN, SECOND_TURN, THIRD_TURN])
        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    @patch("cashier.model.model_util.generate_random_string")
    def test_add_user_turn_handle_wait(
        self,
        generate_random_string_patch,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
    ):
        generate_random_string_patch.return_value = "call_123"
        self.add_user_turn(agent_executor, "hello", model_provider, False, 2)

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")

        fake_fn_call = FunctionCall.create_fake_fn_call(
            model_provider,
            "think",
            args={
                "thought": "At least part of the customer request/question is off-topic for the current conversation and will actually be addressed later. According to the policies, I must tell the customer that 1) their off-topic request/question will be addressed later and 2) we must finish the current business before we can get to it. I must refuse to engage with the off-topic request/question in any way."
            },
        )

        FOURTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=[fake_fn_call],
            fn_call_id_to_fn_output={fake_fn_call.id: None},
        )

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN, FOURTH_TURN]
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def test_add_assistant_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
    ):
        self.add_user_turn(agent_executor, "hello", model_provider, True)
        self.add_assistant_turn(agent_executor, model_provider, "hello back", is_stream)

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")
        FOURTH_TURN = AssistantTurn(
            msg_content="hello back",
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=[],
            fn_call_id_to_fn_output={},
        )

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN, FOURTH_TURN]
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def test_add_assistant_turn_tool_calls(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        fn_names,
        agent_executor,
    ):
        self.add_user_turn(agent_executor, "hello", model_provider, True)
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            model_provider, fn_names, agent_executor.curr_node
        )
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            fn_calls,
            fn_call_id_to_fn_output,
        )

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")
        FOURTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_call_id_to_fn_output,
        )

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN, FOURTH_TURN]
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    @pytest.mark.parametrize(
        "other_fn_names",
        [
            [],
            ["get_menu_item_from_name"],
            ["get_state"],
            ["inexistent_fn"],
            ["get_menu_item_from_name", "get_menu_item_from_name"],
            ["get_state", "inexistent_fn"],
            ["get_state", "get_menu_item_from_name"],
            [
                "get_state",
                "get_menu_item_from_name",
                "get_menu_item_from_name",
            ],
        ],
    )
    def test_add_assistant_state_update_before_user_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        other_fn_names,
        agent_executor,
    ):
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            model_provider, other_fn_names, agent_executor.curr_node
        )
        fn_call = FunctionCall.create_fake_fn_call(
            model_provider, name="update_state_order", args={"order": None}
        )
        fn_calls.append(fn_call)
        fn_call_id_to_fn_output[fn_call.id] = ToolExceptionWrapper(
            StateUpdateError(
                "cannot update any state field until you get the first customer message in the current conversation. Remember, the current conversation starts after <cutoff_msg>"
            )
        )

        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            fn_calls,
            fn_call_id_to_fn_output,
        )

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=agent_executor.curr_node.id,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_call_id_to_fn_output,
        )

        TC = self.create_turn_container([FIRST_TURN, SECOND_TURN, THIRD_TURN])

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def test_node_transition(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        fn_names,
        agent_executor,
    ):
        self.add_user_turn(agent_executor, "hello", model_provider, True)
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            model_provider, fn_names, agent_executor.curr_node
        )
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            fn_calls,
            fn_call_id_to_fn_output,
        )
        self.add_user_turn(agent_executor, "i want pecan latte", model_provider, True)

        order = Order(
            item_orders=[ItemOrder(name="pecan latte", size=CupSize.VENTI, options=[])]
        )
        fn_call_1 = FunctionCall.create_fake_fn_call(
            model_provider,
            name="update_state_order",
            args={"order": order.model_dump()},
        )
        fn_call_2 = FunctionCall.create_fake_fn_call(
            model_provider,
            name="update_state_has_finished_ordering",
            args={"has_finished_ordering": True},
        )
        second_fn_calls = [fn_call_1, fn_call_2]
        second_fn_call_id_to_fn_output = {
            fn_call.id: None for fn_call in second_fn_calls
        }
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            second_fn_calls,
            second_fn_call_id_to_fn_output,
        )

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=1,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")
        FOURTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_call_id_to_fn_output,
        )
        FIFTH_TURN = UserTurn(msg_content="i want pecan latte")
        SIXTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=second_fn_calls,
            fn_call_id_to_fn_output=second_fn_call_id_to_fn_output,
        )

        next_node_schema = cashier_graph_schema.from_node_schema_id_to_edge_schema[
            self.start_node_schema.id
        ][0].to_node_schema
        SEVENTH_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_node_schema.node_system_prompt(
                    node_prompt=next_node_schema.node_prompt,
                    input=order.model_dump_json(),
                    node_input_json_schema=next_node_schema.input_pydantic_model.model_json_schema(),
                    state_json_schema=next_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg="i want pecan latte",
                ),
                node_id=2,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )

        TC = self.create_turn_container(
            [
                FIRST_TURN,
                SECOND_TURN,
                THIRD_TURN,
                FOURTH_TURN,
                FIFTH_TURN,
                SIXTH_TURN,
                SEVENTH_TURN,
            ]
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": next_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def test_backward_node_transition(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        fn_names,
        agent_executor,
    ):

        self.add_user_turn(agent_executor, "hello", model_provider, True)
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            model_provider, fn_names, agent_executor.curr_node
        )
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            fn_calls,
            fn_call_id_to_fn_output,
        )
        self.add_user_turn(agent_executor, "i want pecan latte", model_provider, True)

        order = Order(
            item_orders=[ItemOrder(name="pecan latte", size=CupSize.VENTI, options=[])]
        )
        fn_call_1 = FunctionCall.create_fake_fn_call(
            model_provider,
            name="update_state_order",
            args={"order": order.model_dump()},
        )
        fn_call_2 = FunctionCall.create_fake_fn_call(
            model_provider,
            name="update_state_has_finished_ordering",
            args={"has_finished_ordering": True},
        )
        second_fn_calls = [fn_call_1, fn_call_2]
        second_fn_call_id_to_fn_output = {
            fn_call.id: None for fn_call in second_fn_calls
        }
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            second_fn_calls,
            second_fn_call_id_to_fn_output,
        )
        self.add_assistant_turn(
            agent_executor,
            model_provider,
            "can you confirm the order?",
            is_stream,
        )

        with patch(
            "cashier.model.model_util.FunctionCall.create_fake_fn_call"
        ) as generate_random_string_patch:
            get_state_fn_call = FunctionCall(
                name="get_state",
                id="call_123",
                args={},
            )
            generate_random_string_patch.return_value = get_state_fn_call
            self.add_user_turn(
                agent_executor,
                "i want to change order",
                model_provider,
                False,
                bwd_skip_node_schema_id=self.start_node_schema.id,
            )

        FIRST_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg=None,
                ),
                node_id=1,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")
        FOURTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_call_id_to_fn_output,
        )
        FIFTH_TURN = UserTurn(msg_content="i want pecan latte")
        SIXTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=second_fn_calls,
            fn_call_id_to_fn_output=second_fn_call_id_to_fn_output,
        )

        next_node_schema = cashier_graph_schema.from_node_schema_id_to_edge_schema[
            self.start_node_schema.id
        ][0].to_node_schema
        SEVENTH_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=next_node_schema.node_system_prompt(
                    node_prompt=next_node_schema.node_prompt,
                    input=order.model_dump_json(),
                    node_input_json_schema=next_node_schema.input_pydantic_model.model_json_schema(),
                    state_json_schema=next_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg="i want pecan latte",
                ),
                node_id=2,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls},
        )
        EIGHTH_TURN = AssistantTurn(
            msg_content="can you confirm the order?",
            model_provider=model_provider,
            tool_registry=next_node_schema.tool_registry,
            fn_calls=[],
            fn_call_id_to_fn_output={},
        )
        NINTH_TURN = UserTurn(
            msg_content="i want to change order",
        )
        TENTH_TURN = TurnArgs(
            turn=NodeSystemTurn(
                msg_content=self.start_node_schema.node_system_prompt(
                    node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                    input=None,
                    node_input_json_schema=None,
                    state_json_schema=self.start_node_schema.state_pydantic_model.model_json_schema(),
                    last_msg="can you confirm the order?",
                ),
                node_id=3,
            ),
            kwargs={"remove_prev_tool_calls": remove_prev_tool_calls, "is_skip": True},
        )

        ELEVENTH_TURN = AssistantTurn(
            msg_content=None,
            model_provider=model_provider,
            tool_registry=self.start_node_schema.tool_registry,
            fn_calls=[get_state_fn_call],
            fn_call_id_to_fn_output={
                get_state_fn_call.id: agent_executor.curr_node.state
            },
        )

        TC = self.create_turn_container(
            [
                FIRST_TURN,
                SECOND_TURN,
                THIRD_TURN,
                FOURTH_TURN,
                FIFTH_TURN,
                SIXTH_TURN,
                SEVENTH_TURN,
                EIGHTH_TURN,
                NINTH_TURN,
                TENTH_TURN,
                ELEVENTH_TURN,
            ],
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": self.start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )
