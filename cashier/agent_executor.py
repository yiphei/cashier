import json
from typing import Any, Callable, Dict, Optional, Tuple, cast

from colorama import Style

from cashier.audio import get_speech_from_text
from cashier.graph.request_graph import RequestGraph
from cashier.gui import MessageDisplay
from cashier.logger import logger
from cashier.model.model_completion import ModelOutput
from cashier.model.model_util import CustomJSONEncoder, FunctionCall, ModelProvider
from cashier.tool.function_call_context import (
    FunctionCallContext,
    InexistentFunctionError,
)
from cashier.turn_container import TurnContainer


class AgentExecutor:

    def __init__(
        self,
        audio_output: bool,
        remove_prev_tool_calls: bool,
        graph_schema=None,
    ):
        self.audio_output = audio_output
        self.last_model_provider = None
        self.TC = TurnContainer(remove_prev_tool_calls=remove_prev_tool_calls)

        self.need_user_input = True

        self.graph = RequestGraph(None, graph_schema)
        self.graph.init_next_node(
            graph_schema.start_node_schema,
            None,
            self.TC,
        )
        self.force_tool_choice = None

    def add_user_turn(
        self, msg: str, model_provider: Optional[ModelProvider] = None
    ) -> None:
        MessageDisplay.print_msg("user", msg)
        self.TC.add_user_turn(msg)
        model_provider = (
            model_provider or self.last_model_provider or ModelProvider.OPENAI
        )
        self.graph.handle_user_turn(msg, self.TC, model_provider)

    def add_assistant_turn(
        self, model_completion: ModelOutput, fn_callback: Optional[Callable] = None
    ) -> None:
        self.last_model_provider = model_completion.model_provider
        message = model_completion.get_or_stream_message()
        if message is not None:
            if self.audio_output:
                get_speech_from_text(message)
                MessageDisplay.display_assistant_message(
                    cast(str, model_completion.msg_content)
                )
            else:
                MessageDisplay.display_assistant_message(message)
            self.need_user_input = True

        current_node = self.graph.curr_conversation_node
        fn_calls, fn_id_to_output = self.graph.handle_assistant_turn(model_completion, self.TC, fn_callback)
        if fn_calls:
            self.need_user_input = False

        self.TC.add_assistant_turn(
            model_completion.msg_content,
            model_completion.model_provider,
            current_node.schema.tool_registry,
            fn_calls,
            fn_id_to_output,
        )

    def get_model_completion_kwargs(self) -> Dict[str, Any]:
        force_tool_choice = self.force_tool_choice
        self.force_tool_choice = None
        return {
            "turn_container": self.TC,
            "tool_registry": (self.graph.curr_conversation_node.schema.tool_registry),
            "force_tool_choice": force_tool_choice,
            "exclude_update_state_fns": (
                not self.graph.curr_conversation_node.first_user_message
            ),
        }
