from __future__ import annotations

from typing import Any, List, Optional, Type, Union

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.edge_schema import EdgeSchema
from cashier.graph.new_classes import ActionableMixin, ActionableSchemaMixin, Direction
from cashier.graph.state_model import BaseStateModel
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.tool_registry import ToolRegistry


class Node(ActionableMixin):
    _counter = 0

    def __init__(
        self,
        schema: ActionableSchemaMixin,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        in_edge_schema: Optional[EdgeSchema],
        direction: Direction = Direction.FWD,
    ):
        Node._counter += 1
        self.id = Node._counter
        super().__init__(
            schema,
            input,
            state,
            prompt,
            in_edge_schema,
            direction,
        )


class NodeSchema(ActionableSchemaMixin):
    _counter = 0
    instance_cls = None

    def __init__(
        self,
        node_prompt: str,
        node_system_prompt: Type[NodeSystemPrompt],
        input_pydantic_model: Optional[Type[BaseModel]] = None,
        state_pydantic_model: Optional[Type[BaseStateModel]] = None,
        tool_registry_or_tool_defs: Optional[
            Union[ToolRegistry, List[ChatCompletionToolParam]]
        ] = None,
        first_turn: Optional[ModelTurn] = None,
        run_assistant_turn_before_transition: bool = False,
        tool_names: Optional[List[str]] = None,
    ):
        NodeSchema._counter += 1
        self.id = NodeSchema._counter

        super().__init__(
            node_prompt,
            node_system_prompt,
            input_pydantic_model,
            state_pydantic_model,
            tool_registry_or_tool_defs,
            first_turn,
            run_assistant_turn_before_transition,
            tool_names,
        )
