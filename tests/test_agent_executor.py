from io import StringIO
from unittest.mock import Mock, patch

import pytest
from deepdiff import DeepDiff

from cashier.agent_executor import AgentExecutor
from cashier.graph import Node
from cashier.graph_data.cashier import cashier_graph_schema
from cashier.model import AnthropicModelOutput, Model, OAIModelOutput
from cashier.model_turn import AssistantTurn, NodeSystemTurn, UserTurn
from cashier.model_util import FunctionCall, ModelProvider
from cashier.turn_container import TurnContainer


class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = Mock(spec=Model)
        self.stdout_patcher = patch("sys.stdout", new_callable=StringIO)
        self.stdout_patcher.start()
        Node._counter = 0

        yield

        self.stdout_patcher.stop()
        self.model.reset_mock()

    def create_turn_container(self, turns, remove_prev_tool_calls):
        TC = TurnContainer()
        for turn in turns:
            add_fn = None
            kwargs = {"turn": turn}
            if isinstance(turn, NodeSystemTurn):
                add_fn = "add_node_turn"
                kwargs = {
                    **kwargs,
                    "remove_prev_tool_calls": remove_prev_tool_calls,
                    "is_skip": False,
                }
            elif isinstance(turn, AssistantTurn):
                add_fn = "add_assistant_turn"
            elif isinstance(turn, UserTurn):
                add_fn = "add_user_turn"

            for mm in TC.model_provider_to_message_manager.values():
                getattr(mm, add_fn)(**kwargs)

            TC.turns.append(turn)
        return TC

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

    @pytest.mark.parametrize(
        "model_provider", [ModelProvider.ANTHROPIC, ModelProvider.OPENAI]
    )
    @pytest.mark.parametrize("remove_prev_tool_calls", [True, False])
    def test_initial_node(self, remove_prev_tool_calls, agent_executor):
        start_node_schema = cashier_graph_schema.start_node_schema
        FIRST_TURN = NodeSystemTurn(
            msg_content=start_node_schema.node_system_prompt(
                node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                input=None,
                node_input_json_schema=None,
                state_json_schema=start_node_schema.state_pydantic_model.model_json_schema(),
                last_msg=None,
            ),
            node_id=1,
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN], remove_prev_tool_calls
        )
        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    @pytest.mark.parametrize(
        "model_provider", [ModelProvider.ANTHROPIC, ModelProvider.OPENAI]
    )
    @pytest.mark.parametrize("remove_prev_tool_calls", [True, False])
    def test_add_user_turn(
        self, model_provider, remove_prev_tool_calls, agent_executor
    ):
        is_on_topic_model_completion = Mock(
            spec=(
                OAIModelOutput
                if model_provider == ModelProvider.OPENAI
                else AnthropicModelOutput
            )
        )
        is_on_topic_model_completion.get_message_prop.return_value = True
        if model_provider == ModelProvider.OPENAI:
            is_on_topic_model_completion.get_prob.return_value = 0.5
        self.model.chat.return_value = is_on_topic_model_completion

        agent_executor.add_user_turn("hello")

        start_node_schema = cashier_graph_schema.start_node_schema
        FIRST_TURN = NodeSystemTurn(
            msg_content=start_node_schema.node_system_prompt(
                node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                input=None,
                node_input_json_schema=None,
                state_json_schema=start_node_schema.state_pydantic_model.model_json_schema(),
                last_msg=None,
            ),
            node_id=1,
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN], remove_prev_tool_calls
        )
        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    @pytest.mark.parametrize(
        "model_provider", [ModelProvider.ANTHROPIC, ModelProvider.OPENAI]
    )
    @pytest.mark.parametrize("remove_prev_tool_calls", [True, False])
    @patch("cashier.model_util.generate_random_string")
    def test_add_user_turn_handle_wait(
        self,
        generate_random_string_patch,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
    ):
        generate_random_string_patch.return_value = "call_123"
        is_on_topic_model_completion = Mock(
            spec=(
                OAIModelOutput
                if model_provider == ModelProvider.OPENAI
                else AnthropicModelOutput
            )
        )
        is_on_topic_model_completion.get_message_prop.return_value = False
        if model_provider == ModelProvider.OPENAI:
            is_on_topic_model_completion.get_prob.return_value = 0.5

        is_wait_model_completion = Mock(
            spec=(
                OAIModelOutput
                if model_provider == ModelProvider.OPENAI
                else AnthropicModelOutput
            )
        )
        is_wait_model_completion.get_message_prop.return_value = 2
        if model_provider == ModelProvider.OPENAI:
            is_wait_model_completion.get_prob.return_value = 0.5

        self.model.chat.side_effect = [
            is_on_topic_model_completion,
            is_wait_model_completion,
        ]

        agent_executor.add_user_turn("hello")

        start_node_schema = cashier_graph_schema.start_node_schema
        FIRST_TURN = NodeSystemTurn(
            msg_content=start_node_schema.node_system_prompt(
                node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                input=None,
                node_input_json_schema=None,
                state_json_schema=start_node_schema.state_pydantic_model.model_json_schema(),
                last_msg=None,
            ),
            node_id=1,
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
            tool_registry=start_node_schema.tool_registry,
            fn_calls=[fake_fn_call],
            fn_call_id_to_fn_output={fake_fn_call.id: None},
        )

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN, FOURTH_TURN], remove_prev_tool_calls
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )

    def create_mock_model_completion(self, model_provider, is_stream, message):
        model_completion_class = (
            OAIModelOutput
            if model_provider == ModelProvider.OPENAI
            else AnthropicModelOutput
        )

        model_completion = model_completion_class(output_obj=None, is_stream=is_stream)
        model_completion.msg_content = message
        model_completion.get_message = Mock(return_value=message)
        model_completion.stream_message = Mock(return_value=iter(message.split(" ")))
        model_completion.get_fn_calls = Mock(return_value=[])
        model_completion.stream_fn_calls = Mock(return_value=iter([]))
        return model_completion

    @pytest.mark.parametrize(
        "model_provider", [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]
    )
    @pytest.mark.parametrize("remove_prev_tool_calls", [True, False])
    @pytest.mark.parametrize("is_stream", [True, False])
    def test_add_assistant_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
    ):
        is_on_topic_model_completion = Mock(
            spec=(
                OAIModelOutput
                if model_provider == ModelProvider.OPENAI
                else AnthropicModelOutput
            )
        )
        is_on_topic_model_completion.get_message_prop.return_value = True
        if model_provider == ModelProvider.OPENAI:
            is_on_topic_model_completion.get_prob.return_value = 0.5

        self.model.chat.return_value = is_on_topic_model_completion

        agent_executor.add_user_turn("hello")

        model_completion = self.create_mock_model_completion(
            model_provider, is_stream, "hello back"
        )

        agent_executor.add_assistant_turn(model_completion)

        start_node_schema = cashier_graph_schema.start_node_schema
        FIRST_TURN = NodeSystemTurn(
            msg_content=start_node_schema.node_system_prompt(
                node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                input=None,
                node_input_json_schema=None,
                state_json_schema=start_node_schema.state_pydantic_model.model_json_schema(),
                last_msg=None,
            ),
            node_id=1,
        )
        SECOND_TURN = cashier_graph_schema.start_node_schema.first_turn
        THIRD_TURN = UserTurn(msg_content="hello")
        FOURTH_TURN = AssistantTurn(
            msg_content="hello back",
            model_provider=model_provider,
            tool_registry=start_node_schema.tool_registry,
            fn_calls=[],
            fn_call_id_to_fn_output={},
        )

        TC = self.create_turn_container(
            [FIRST_TURN, SECOND_TURN, THIRD_TURN, FOURTH_TURN], remove_prev_tool_calls
        )

        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": start_node_schema.tool_registry,
                "force_tool_choice": None,
            },
        )
