from __future__ import annotations

from enum import StrEnum
from inspect import signature
from typing import Any, List, Literal, Optional, Type, Union, cast, overload

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel

from cashier.graph.edge_schema import (
    BwdStateInit,
    EdgeSchema,
    FwdStateInit,
)
from cashier.graph.state import BaseStateModel, HasStateMixin, HasStateSchemaMixin
from cashier.model.model_turn import ModelTurn
from cashier.prompts.node_system import NodeSystemPrompt
from cashier.tool.tool_registry import ToolRegistry


class AutoMixinInit(type):
    """Metaclass that automatically initializes mixins in the correct order."""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)

        # Get all base classes that end with 'Mixin'
        mixins = [base for base in cls.__bases__ if base.__name__.endswith("Mixin")]

        # Initialize each mixin with matching kwargs
        for mixin in mixins:
            # Get the init parameters for this mixin
            if hasattr(mixin, "__init__"):
                # Get only the parameter names from the function signature
                init_params = list(signature(mixin.__init__).parameters.keys())[
                    1:
                ]  # Skip 'self'

                # Filter kwargs to only include parameters that match this mixin's init
                mixin_kwargs = {k: v for k, v in kwargs.items() if k in init_params}

                # Call the mixin's init
                mixin.__init__(instance, **mixin_kwargs)

        if "__init__" in cls.__dict__:
            cls.__init__(instance, *args, **kwargs)
        return instance


class HasIdMixin:
    _counter = 0

    def __init__(self):
        self.__class__._counter += 1
        self.id = self.__class__._counter


class Direction(StrEnum):
    FWD = "FWD"
    BWD = "BWD"


class HasActionableSchemaMixin(HasStateSchemaMixin):
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
        super().__init__(state_pydantic_model)
        self.node_prompt = node_prompt
        self.node_system_prompt = node_system_prompt
        self.input_pydantic_model = input_pydantic_model
        self.first_turn = first_turn
        self.run_assistant_turn_before_transition = run_assistant_turn_before_transition
        if tool_registry_or_tool_defs is not None and isinstance(
            tool_registry_or_tool_defs, ToolRegistry
        ):
            self.tool_registry = (
                tool_registry_or_tool_defs.__class__.create_from_tool_registry(
                    tool_registry_or_tool_defs, tool_names
                )
            )
        else:
            self.tool_registry = ToolRegistry(tool_registry_or_tool_defs)

        if self.state_pydantic_model is not None:
            for (
                field_name,
                field_info,
            ) in self.state_pydantic_model.model_fields.items():
                new_tool_fn_name = f"update_state_{field_name}"
                field_args = {field_name: (field_info.annotation, field_info)}
                self.tool_registry.add_tool_def(
                    new_tool_fn_name,
                    f"Function to update the `{field_name}` field in the state",
                    field_args,
                )

            self.tool_registry.add_tool_def(
                "get_state",
                "Function to get the current state, as defined in <state>",
                {},
            )

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: Literal[None] = None,
        edge_schema: Literal[None] = None,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasActionableMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: Literal[None] = None,
        direction: Literal[Direction.FWD] = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasActionableMixin: ...

    @overload
    def create_node(  # noqa: E704
        self,
        input: Any,
        last_msg: str,
        edge_schema: EdgeSchema,
        prev_node: HasActionableMixin,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasActionableMixin: ...

    def create_node(
        self,
        input: Any,
        last_msg: Optional[str] = None,
        edge_schema: Optional[EdgeSchema] = None,
        prev_node: Optional[HasActionableMixin] = None,
        direction: Direction = Direction.FWD,
        curr_request: Optional[str] = None,
    ) -> HasActionableMixin:
        state = HasActionableMixin.init_state(
            self.state_pydantic_model, prev_node, edge_schema, direction, input
        )

        prompt = self.node_system_prompt(
            node_prompt=self.node_prompt,
            input=(
                input.model_dump_json()
                if self.input_pydantic_model is not None
                else None
            ),
            node_input_json_schema=(
                self.input_pydantic_model.model_json_schema()
                if self.input_pydantic_model
                else None
            ),
            state_json_schema=(
                self.state_pydantic_model.model_json_schema()
                if self.state_pydantic_model
                else None
            ),
            last_msg=last_msg,
            curr_request=curr_request,
        )

        if direction == Direction.BWD:
            assert prev_node is not None
            in_edge_schema = prev_node.in_edge_schema
        else:
            in_edge_schema = edge_schema
        return self.instance_cls(
            schema=self,
            input=input,
            state=state,
            prompt=cast(str, prompt),
            in_edge_schema=in_edge_schema,
            direction=direction,
        )


class HasActionableMixin(HasStateMixin):
    class Status(StrEnum):
        IN_PROGRESS = "IN_PROGRESS"
        COMPLETED = "COMPLETED"

    def __init__(
        self,
        schema: HasActionableSchemaMixin,
        input: Any,
        state: BaseStateModel,
        prompt: str,
        in_edge_schema: Optional[EdgeSchema],
        direction: Direction = Direction.FWD,
    ):
        super().__init__(state)
        self.prompt = prompt
        self.input = input
        self.schema = schema
        self.status = self.Status.IN_PROGRESS
        self.in_edge_schema = in_edge_schema
        self.direction = direction
        self.has_run_assistant_turn_before_transition = False

    @classmethod
    def init_state(
        cls,
        state_pydantic_model: Optional[Type[BaseStateModel]],
        prev_node: Optional[HasActionableMixin],
        edge_schema: Optional[EdgeSchema],
        direction: Direction,
        input: Any,
    ) -> Optional[BaseStateModel]:
        if state_pydantic_model is None:
            return None

        if prev_node is not None:
            state_init_val = getattr(
                edge_schema,
                "fwd_state_init" if direction == Direction.FWD else "bwd_state_init",
            )
            state_init_enum_cls = (
                FwdStateInit if direction == Direction.FWD else BwdStateInit
            )

            if state_init_val == state_init_enum_cls.RESET:  # type: ignore
                return state_pydantic_model()
            elif state_init_val == state_init_enum_cls.RESUME or (  # type: ignore
                direction == Direction.FWD
                and state_init_val == state_init_enum_cls.RESUME_IF_INPUT_UNCHANGED  # type: ignore
                and input == prev_node.input
            ):
                return prev_node.state.copy_resume()

        return state_pydantic_model()

    def mark_as_completed(self) -> None:
        self.status = self.Status.COMPLETED


