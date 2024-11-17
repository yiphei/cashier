from io import StringIO
from unittest.mock import Mock, patch

import pytest

from cashier.agent_executor import AgentExecutor
from cashier.graph import Node
from cashier.graph_data.cashier import cashier_graph_schema
from cashier.model import Model
from cashier.model_turn import AssistantTurn, NodeSystemTurn, UserTurn
from cashier.model_util import ModelProvider
from cashier.turn_container import TurnContainer
from deepdiff import DeepDiff


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
                kwargs = { **kwargs,
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
        FIRST_NODE = NodeSystemTurn(
            msg_content=start_node_schema.node_system_prompt(
                node_prompt=cashier_graph_schema.start_node_schema.node_prompt,
                input=None,
                node_input_json_schema=None,
                state_json_schema=start_node_schema.state_pydantic_model.model_json_schema(),
                last_msg=None,
            ),
            node_id=1,
        )
        SECOND_NODE = cashier_graph_schema.start_node_schema.first_turn

        TC = self.create_turn_container([FIRST_NODE, SECOND_NODE], remove_prev_tool_calls)
        assert not DeepDiff(agent_executor.get_model_completion_kwargs(), {
        "turn_container": TC,
            "tool_registry": start_node_schema.tool_registry,
            "force_tool_choice": None,
        })
