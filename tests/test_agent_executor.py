import unittest
from cashier.agent_executor import AgentExecutor
from cashier.model import Model
from cashier.graph_data.cashier import cashier_graph_schema
from cashier.model_util import ModelProvider
from cashier.model_turn import NodeSystemTurn

class Agent(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        print("AAAA")

    def _init_agent_executor(self, remove_prev_tool_calls, model_provider):
        return AgentExecutor(
            model= self.model,
            elevenlabs_client=None,
            graph_schema=cashier_graph_schema,
            audio_output=False,
            remove_prev_tool_calls=remove_prev_tool_calls,
            model_provider= model_provider,
        )
    
    def test_initial_node(self):
        AE = self._init_agent_executor(True, ModelProvider.ANTHROPIC)

        FIRST_NODE = NodeSystemTurn(msg_content=cashier_graph_schema.start_node_schema.node_system_prompt(), node_id=1)
        SECOND_NODE = cashier_graph_schema.start_node_schema.first_turn

        self.assertEqual(AE.TC, [FIRST_NODE, SECOND_NODE])
