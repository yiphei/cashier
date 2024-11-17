from io import StringIO
from unittest.mock import Mock, patch

import pytest

from cashier.agent_executor import AgentExecutor
from cashier.graph import Node
from cashier.graph_data.cashier import cashier_graph_schema
from cashier.model import Model
from cashier.model_turn import NodeSystemTurn
from cashier.model_util import ModelProvider


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
    def test_initial_node(self, agent_executor):
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
        assert agent_executor.TC.turns == [FIRST_NODE, SECOND_NODE]
