import os
import uuid
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
from typing import Any, Dict
from unittest.mock import Mock, call, patch

import pytest
from deepdiff import DeepDiff
from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel, Field

from cashier.agent_executor import AgentExecutor
from cashier.model.message_list import MessageList
from cashier.model.model_completion import AnthropicModelOutput, OAIModelOutput
from cashier.model.model_turn import (
    AssistantTurn,
    ModelTurn,
    NodeSystemTurn,
    SystemTurn,
    UserTurn,
)
from cashier.model.model_util import (
    MODEL_PROVIDER_TO_TOOL_CALL_ID_PREFIX,
    FunctionCall,
    ModelProvider,
    generate_random_string,
)
from cashier.prompts.graph_schema_selection import AgentSelection
from cashier.tool.function_call_context import (
    InexistentFunctionError,
    StateUpdateError,
    ToolExceptionWrapper,
)
from cashier.turn_container import TurnContainer
from data.graph.airline import (
    AIRLINE_REQUEST_SCHEMA,
    BOOK_FLIGHT_GRAPH_SCHEMA,
    find_flight_node_schema,
    get_user_id_node_schema,
)
from data.types.airline import FlightInfo, UserDetails


class TurnArgs(BaseModel):
    turn: ModelTurn
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TestAirline:
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

        self.rand_tool_ids = deque()
        self.rand_uuids = deque()
        self.model_chat_patcher = patch("cashier.model.model_completion.Model.chat")
        self.model_chat = self.model_chat_patcher.start()

        yield

        self.rand_tool_ids.clear()
        self.rand_uuids.clear()
        self.model_chat_patcher.stop()

    def get_next_conv_node_schema(self, curr_node_schema):
        edge_schema = self.graph_schema.from_conv_node_schema_id_to_edge_schema[
            curr_node_schema.id
        ]
        return self.edge_schema_id_to_to_cov_node_schema_id[edge_schema.id]

    @contextmanager
    def generate_random_string_context(self):
        original_generate_random_string = generate_random_string
        original_uuid4 = uuid.uuid4

        def capture_fn_call(*args, **kwargs):
            output = original_generate_random_string(*args, **kwargs)
            self.rand_tool_ids.append(output)
            return output

        def capture_uuid_call(*args, **kwargs):
            output = original_uuid4(*args, **kwargs)
            self.rand_uuids.append(str(output))
            return output

        with patch(
            "cashier.model.model_util.generate_random_string",
            side_effect=capture_fn_call,
        ), patch("cashier.model.model_util.uuid.uuid4", side_effect=capture_uuid_call):
            yield

    def create_turn_container(self, turn_args_list, remove_prev_tool_calls):
        TC = TurnContainer(remove_prev_tool_calls=remove_prev_tool_calls)
        for turn_args in turn_args_list:
            add_fn = None
            if isinstance(turn_args, TurnArgs):
                turn = turn_args.turn
                kwargs = {
                    "turn": turn_args.turn,
                    "remove_prev_tool_calls": remove_prev_tool_calls,
                    **turn_args.kwargs,
                }
            else:
                turn = turn_args
                kwargs = {"turn": turn_args}

            if isinstance(turn, NodeSystemTurn):
                add_fn = "add_node_turn"
            elif isinstance(turn, AssistantTurn):
                add_fn = "add_assistant_turn"
            elif isinstance(turn, UserTurn):
                add_fn = "add_user_turn"
            elif isinstance(turn, SystemTurn):
                add_fn = "add_system_turn"

            for mm in TC.model_provider_to_message_manager.values():
                getattr(mm, add_fn)(**kwargs)

            TC.turns.append(turn)
        return TC

    def run_message_dict_assertions(self, agent_executor, model_provider):
        assert not DeepDiff(
            self.message_dicts,
            agent_executor.TC.model_provider_to_message_manager[
                model_provider
            ].message_dicts,
        )
        assert not DeepDiff(
            self.conversation_dicts,
            agent_executor.TC.model_provider_to_message_manager[
                model_provider
            ].conversation_dicts,
        )
        assert not DeepDiff(
            self.node_conversation_dicts,
            agent_executor.TC.model_provider_to_message_manager[
                model_provider
            ].node_conversation_dicts,
        )

    def run_assertions(self, agent_executor, TC, tool_registry, model_provider):
        self.run_message_dict_assertions(agent_executor, model_provider)
        assert not DeepDiff(
            agent_executor.get_model_completion_kwargs(),
            {
                "turn_container": TC,
                "tool_registry": tool_registry,
                "force_tool_choice": None,
                "exclude_update_state_fns": (
                    not agent_executor.graph.curr_conversation_node.first_user_message
                    if agent_executor.graph.curr_conversation_node is not None
                    else False
                ),
            },
            exclude_regex_paths=r"root\['turn_container'\]\.turns\[\d+\]\.node_id",
        )

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
            if message_prop == "null":  # TODO: fix this
                message_prop = None
            model_completion.get_message_prop = Mock(return_value=message_prop)
            if model_provider == ModelProvider.OPENAI:
                model_completion.get_prob = Mock(return_value=prob)
        return model_completion

    def recreate_fake_single_fn_call(self, fn_name, args):
        id = self.rand_uuids.popleft()
        oai_api_id = (
            MODEL_PROVIDER_TO_TOOL_CALL_ID_PREFIX[ModelProvider.OPENAI]
            + self.rand_tool_ids.popleft()
        )
        anthropic_api_id = (
            MODEL_PROVIDER_TO_TOOL_CALL_ID_PREFIX[ModelProvider.ANTHROPIC]
            + self.rand_tool_ids.popleft()
        )
        fn = FunctionCall(
            id=id,
            name=fn_name,
            oai_api_id=oai_api_id,
            anthropic_api_id=anthropic_api_id,
            api_id_model_provider=None,
            args=args,
        )
        fn.model_fields_set.remove("id")
        return fn

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
                model_fields = node.schema.state_schema.model_fields
                field_info = model_fields[field_name]

                # Get default value or call default_factory if it exists
                default_value = (
                    field_info.default_factory()
                    if field_info.default_factory is not None
                    else field_info.default
                )
                args = {field_name: default_value}

            fn_call = FunctionCall.create(
                api_id_model_provider=model_provider,
                api_id=FunctionCall.generate_fake_id(model_provider),
                name=fn_name,
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
        wait_node_schema_id=None,
        skip_node_schema_id=None,
    ):
        model_chat_side_effects = []

        is_on_topic_model_completion = self.create_mock_model_completion(
            model_provider, None, False, is_on_topic, 0.5
        )
        model_chat_side_effects.append(is_on_topic_model_completion)
        if not is_on_topic:
            agent_addition_completion = self.create_mock_model_completion(
                model_provider, None, False, "null", 0.5
            )
            model_chat_side_effects.append(agent_addition_completion)

            is_wait_model_completion = self.create_mock_model_completion(
                model_provider,
                None,
                False,
                wait_node_schema_id
                or agent_executor.graph.curr_conversation_node.schema.id,
                0.5,
            )
            model_chat_side_effects.append(is_wait_model_completion)

            if wait_node_schema_id is None:
                skip_model_completion = self.create_mock_model_completion(
                    model_provider,
                    None,
                    False,
                    skip_node_schema_id
                    or agent_executor.graph.curr_conversation_node.schema.id,
                    0.5,
                )
                model_chat_side_effects.append(skip_model_completion)

        self.model_chat.side_effect = model_chat_side_effects
        with self.generate_random_string_context():
            agent_executor.add_user_turn(message, model_provider)

        ut = UserTurn(msg_content=message)
        self.build_messages_from_turn(ut, model_provider)
        return ut

    def add_request_user_turn(
        self,
        agent_executor,
        message,
        model_provider,
        task,
    ):
        agent_selection = AgentSelection(agent_id=self.graph_schema.id, task=task)
        graph_schema_selection_completion = self.create_mock_model_completion(
            model_provider, None, False, [agent_selection], 0.5
        )
        self.model_chat.side_effect = [graph_schema_selection_completion]
        with self.generate_random_string_context():
            agent_executor.add_user_turn(message, model_provider)

        ut = UserTurn(msg_content=message)
        return ut

    def add_assistant_turn(
        self,
        agent_executor,
        model_provider,
        message,
        is_stream,
        fn_calls=None,
        fn_call_id_to_fn_output=None,
        tool_names=None,
    ):
        if tool_names is not None:
            fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
                model_provider, tool_names, agent_executor.graph.curr_conversation_node
            )

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

        tool_registry = agent_executor.graph.curr_conversation_node.schema.tool_registry

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
            curr_node = agent_executor.graph.curr_conversation_node
            if get_state_fn_call is not None:
                stack.enter_context(
                    patch.object(
                        agent_executor.graph.curr_conversation_node,
                        "get_state",
                        wraps=agent_executor.graph.curr_conversation_node.get_state,
                    )
                )

            if update_state_fn_calls:
                stack.enter_context(
                    patch.object(
                        agent_executor.graph.curr_conversation_node,
                        "update_state",
                        wraps=agent_executor.graph.curr_conversation_node.update_state,
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

        at = AssistantTurn(
            msg_content=message,
            model_provider=model_provider,
            tool_registry=tool_registry,
            fn_calls=fn_calls,
            fn_call_id_to_fn_output=fn_call_id_to_fn_output or {},
        )
        self.build_assistant_turn_messages(at, model_provider)
        return at

    @pytest.fixture
    def agent_executor(self, remove_prev_tool_calls):
        return AgentExecutor(
            graph_schema=AIRLINE_REQUEST_SCHEMA,
            audio_output=False,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )

    @pytest.fixture(autouse=True)
    def setup_message_dicts(self, model_provider):
        self.message_dicts = MessageList(model_provider=model_provider)
        self.conversation_dicts = MessageList(model_provider=model_provider)
        self.node_conversation_dicts = MessageList(model_provider=model_provider)
        yield
        self.message_dicts = None
        self.conversation_dicts = None
        self.node_conversation_dicts = None

    @pytest.fixture(autouse=True)
    def setup_start_message_list(
        self, start_turns, setup_message_dicts, model_provider
    ):
        self.build_messages_from_turn(start_turns[1], model_provider)
        self.build_messages_from_turn(start_turns[2], model_provider)

    @pytest.fixture
    def start_turns(self, agent_executor, model_provider):
        ut = self.add_request_user_turn(
            agent_executor,
            "i want to book flight",
            model_provider,
            "customer wants to book flight",
        )
        second_node_schema = self.start_conv_node_schema
        return [
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

    @pytest.fixture
    def first_into_second_conv_turns(
        self,
        agent_executor,
        model_provider,
        is_stream,
        fn_names,
        remove_prev_tool_calls,
    ):
        t1 = self.add_user_turn(agent_executor, "hello", model_provider, True)
        t2 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            tool_names=fn_names,
        )
        t3 = self.add_user_turn(
            agent_executor, "my username is ...", model_provider, True
        )
        self.run_message_dict_assertions(agent_executor, model_provider)

        user_details = ModelFactory.create_factory(UserDetails).build()
        fn_call_1 = FunctionCall.create(
            api_id_model_provider=model_provider,
            api_id=FunctionCall.generate_fake_id(model_provider),
            name="update_state_user_details",
            args={"user_details": user_details.model_dump()},
        )
        second_fn_calls = [fn_call_1]
        second_fn_call_id_to_fn_output = {
            fn_call.id: None for fn_call in second_fn_calls
        }
        t4 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            None,
            is_stream,
            second_fn_calls,
            second_fn_call_id_to_fn_output,
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
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )
        return [t1, t2, t3, t4, node_turn]

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
            ["get_user_details"],
            ["get_state"],
            ["update_state_user_details"],
            ["inexistent_fn"],
            ["get_user_details", "get_user_details"],
            ["get_state", "update_state_user_details"],
            ["get_state", "update_state_user_details", "inexistent_fn"],
            ["get_state", "get_user_details", "update_state_user_details"],
            [
                "get_state",
                "get_user_details",
                "update_state_user_details",
                "get_user_details",
            ],
        ]
    )
    def fn_names(cls, request):
        return request.param

    def build_user_turn_messages(self, user_turn, model_provider):
        self.message_dicts.extend(
            user_turn.build_messages(model_provider), MessageList.ItemType.USER
        )
        self.conversation_dicts.extend(
            user_turn.build_messages(model_provider), MessageList.ItemType.USER
        )
        self.node_conversation_dicts.extend(
            user_turn.build_messages(model_provider), MessageList.ItemType.USER
        )

    def build_assistant_turn_messages(self, assistant_turn, model_provider):
        messages = assistant_turn.build_messages(model_provider)
        if model_provider == ModelProvider.OPENAI:
            for message in messages:
                if message.get("tool_calls", None) is not None:
                    tool_call_id = message["tool_calls"][0]["id"]
                    curr_fn_name = message["tool_calls"][0]["function"]["name"]
                    self.message_dicts.append(
                        message, MessageList.ItemType.TOOL_CALL, tool_call_id
                    )
                elif message["role"] == "tool":
                    tool_call_id = message["tool_call_id"]
                    self.message_dicts.append(
                        message,
                        MessageList.ItemType.TOOL_OUTPUT,
                        MessageList.get_tool_output_uri_from_tool_id(tool_call_id),
                    )
                elif message["role"] == "system" and curr_fn_name is not None:
                    self.message_dicts.remove_by_uri(curr_fn_name, False)
                    self.message_dicts.append(
                        message, MessageList.ItemType.TOOL_OUTPUT_SCHEMA, curr_fn_name
                    )
                    curr_fn_name = None
                else:
                    self.message_dicts.append(message, MessageList.ItemType.ASSISTANT)
                    self.conversation_dicts.append(
                        message, MessageList.ItemType.ASSISTANT
                    )
                    self.node_conversation_dicts.append(
                        message, MessageList.ItemType.ASSISTANT
                    )
        else:
            if len(messages) == 2:
                [message_1, message_2] = messages
            else:
                [message_1] = messages
                message_2 = None

            contents = message_1["content"]
            self.message_dicts.append(message_1)
            has_fn_calls = False
            if type(contents) is list:
                for content in contents:
                    if content["type"] == "tool_use":
                        tool_call_id = content["id"]
                        self.message_dicts.track_idx(
                            MessageList.ItemType.TOOL_CALL, uri=tool_call_id
                        )
                        has_fn_calls = True

            if not has_fn_calls:
                self.message_dicts.track_idx(MessageList.ItemType.ASSISTANT)
                ass_message = {
                    "role": "assistant",
                    "content": assistant_turn.msg_content,
                }
                self.conversation_dicts.append(
                    ass_message, MessageList.ItemType.ASSISTANT
                )
                self.node_conversation_dicts.append(
                    ass_message, MessageList.ItemType.ASSISTANT
                )

            if message_2 is not None:
                self.message_dicts.append(message_2)
                for content in message_2["content"]:
                    if content["type"] == "tool_result":
                        tool_id = content["tool_use_id"]
                        self.message_dicts.track_idx(
                            MessageList.ItemType.TOOL_OUTPUT,
                            uri=MessageList.get_tool_output_uri_from_tool_id(tool_id),
                        )

    def build_node_turn_messages(
        self,
        node_turn,
        model_provider,
        remove_prev_fn_return_schema,
        remove_prev_tool_calls,
        is_skip,
    ):
        if remove_prev_tool_calls:
            assert remove_prev_fn_return_schema is not False

        if remove_prev_fn_return_schema is True or remove_prev_tool_calls:
            self.message_dicts.clear(MessageList.ItemType.TOOL_OUTPUT_SCHEMA)

        if remove_prev_tool_calls:
            self.message_dicts.clear(
                [MessageList.ItemType.TOOL_CALL, MessageList.ItemType.TOOL_OUTPUT]
            )

        if is_skip:
            self.conversation_dicts.track_idx(
                MessageList.ItemType.NODE, len(self.conversation_dicts) - 2
            )
            self.node_conversation_dicts = self.node_conversation_dicts[-1:]
        else:
            self.conversation_dicts.track_idx(MessageList.ItemType.NODE)
            self.node_conversation_dicts.clear()

        if model_provider == ModelProvider.OPENAI:
            self.message_dicts.clear(MessageList.ItemType.NODE)
            [msg] = node_turn.build_oai_messages()
            if is_skip:
                self.message_dicts.insert(
                    len(self.message_dicts) - 1, msg, MessageList.ItemType.NODE
                )
            else:
                self.message_dicts.append(msg, MessageList.ItemType.NODE)
        else:
            self.system = node_turn.msg_content  # TODO: this is currently not used

            if is_skip:
                self.message_dicts.track_idx(
                    MessageList.ItemType.NODE, len(self.message_dicts) - 2
                )
            else:
                self.message_dicts.track_idx(MessageList.ItemType.NODE)

    def build_messages_from_turn(
        self,
        turn,
        model_provider,
        remove_prev_fn_return_schema=None,
        remove_prev_tool_calls=False,
        is_skip=False,
    ):
        if isinstance(turn, TurnArgs):
            turn = turn.turn

        if isinstance(turn, UserTurn):
            self.build_user_turn_messages(turn, model_provider)
        elif isinstance(turn, AssistantTurn):
            self.build_assistant_turn_messages(turn, model_provider)
        elif isinstance(turn, NodeSystemTurn):
            self.build_node_turn_messages(
                turn,
                model_provider,
                remove_prev_fn_return_schema,
                remove_prev_tool_calls,
                is_skip,
            )
        else:
            raise ValueError(f"Unknown turn type: {type(turn)}")

    def test_graph_initialization(
        self, model_provider, remove_prev_tool_calls, agent_executor, start_turns
    ):
        TC = self.create_turn_container(start_turns, remove_prev_tool_calls)
        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_user_turn(
        self, model_provider, remove_prev_tool_calls, agent_executor, start_turns
    ):
        user_turn = self.add_user_turn(agent_executor, "hello", model_provider, True)

        TC = self.create_turn_container(
            [*start_turns, user_turn], remove_prev_tool_calls
        )
        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_user_turn_with_wait(
        self,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
        start_turns,
    ):
        user_turn = self.add_user_turn(
            agent_executor, "hello", model_provider, False, find_flight_node_schema.id
        )

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
        self.build_messages_from_turn(assistant_turn, model_provider)

        TC = self.create_turn_container(
            [*start_turns, user_turn, assistant_turn], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_assistant_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
        start_turns,
    ):
        user_turn = self.add_user_turn(agent_executor, "hello", model_provider, True)
        assistant_turn = self.add_assistant_turn(
            agent_executor, model_provider, "hello back", is_stream
        )

        TC = self.create_turn_container(
            [*start_turns, user_turn, assistant_turn], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_add_assistant_turn_with_tool_calls(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        fn_names,
        agent_executor,
        start_turns,
    ):
        user_turn = self.add_user_turn(agent_executor, "hello", model_provider, True)
        assistant_turn = self.add_assistant_turn(
            agent_executor, model_provider, None, is_stream, tool_names=fn_names
        )

        TC = self.create_turn_container(
            [*start_turns, user_turn, assistant_turn], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    @pytest.mark.parametrize(
        "other_fn_names",
        [
            [],
            ["get_user_details"],
            ["get_state"],
            ["inexistent_fn"],
            ["get_user_details", "get_user_details"],
            ["get_state", "inexistent_fn"],
            ["get_state", "get_user_details"],
            [
                "get_state",
                "get_user_details",
                "get_user_details",
            ],
        ],
    )
    def test_state_update_before_user_turn(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        other_fn_names,
        agent_executor,
        start_turns,
    ):
        fn_calls, fn_call_id_to_fn_output = self.create_fake_fn_calls(
            model_provider, other_fn_names, agent_executor.graph.curr_conversation_node
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
            agent_executor,
            model_provider,
            None,
            is_stream,
            fn_calls,
            fn_call_id_to_fn_output,
        )

        TC = self.create_turn_container(
            [*start_turns, assistant_turn], remove_prev_tool_calls
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_node_transition(
        self,
        model_provider,
        remove_prev_tool_calls,
        agent_executor,
        start_turns,
        first_into_second_conv_turns,
    ):
        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_conv_turns,
            ],
            remove_prev_tool_calls,
        )

        self.run_assertions(
            agent_executor, TC, find_flight_node_schema.tool_registry, model_provider
        )

    def test_backward_node_skip(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
        start_turns,
        first_into_second_conv_turns,
    ):
        t5 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            "what flight do you want?",
            is_stream,
        )
        self.run_message_dict_assertions(agent_executor, model_provider)

        t6 = self.add_user_turn(
            agent_executor,
            "i want to change my user details",
            model_provider,
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
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
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
        self.build_messages_from_turn(t7, model_provider)

        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_conv_turns,
                t5,
                t6,
                node_turn_2,
                t7,
            ],
            remove_prev_tool_calls,
        )

        self.run_assertions(
            agent_executor,
            TC,
            self.start_conv_node_schema.tool_registry,
            model_provider,
        )

    def test_forward_node_skip(
        self,
        model_provider,
        remove_prev_tool_calls,
        is_stream,
        agent_executor,
        start_turns,
        first_into_second_conv_turns,
    ):
        t5 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            "what flight do you want?",
            is_stream,
        )

        t6 = self.add_user_turn(
            agent_executor,
            "i want flight from ... to ... on ...",
            model_provider,
            True,
        )
        self.run_message_dict_assertions(agent_executor, model_provider)

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
            agent_executor,
            model_provider,
            None,
            is_stream,
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
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
        )
        t8 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            "thanks for confirming flights, now lets move on to ...",
            is_stream,
        )
        self.run_message_dict_assertions(agent_executor, model_provider)

        t9 = self.add_user_turn(
            agent_executor,
            "actually, i want to change my user details",
            model_provider,
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
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
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
        self.build_messages_from_turn(t10, model_provider)
        t11 = self.add_assistant_turn(
            agent_executor,
            model_provider,
            "what do you want to change?",
            is_stream,
        )
        self.run_message_dict_assertions(agent_executor, model_provider)

        t12 = self.add_user_turn(
            agent_executor,
            "nvm, nothing",
            model_provider,
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
            model_provider,
            remove_prev_tool_calls=remove_prev_tool_calls,
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
        self.build_messages_from_turn(t13, model_provider)

        TC = self.create_turn_container(
            [
                *start_turns,
                *first_into_second_conv_turns,
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
            remove_prev_tool_calls,
        )

        self.run_assertions(
            agent_executor, TC, find_flight_node_schema.tool_registry, model_provider
        )


def test_class_test_count(request):
    absolute_path = __file__
    relative_path = os.path.relpath(absolute_path, os.getcwd())
    class_nodeid_prefix = f"{relative_path}::{TestAirline.__name__}::"
    class_items = [
        item
        for item in request.session.items
        if item.nodeid.startswith(class_nodeid_prefix)
    ]

    expected = 372
    actual = len(class_items)

    assert (
        actual == expected
    ), f"Expected {expected} tests in {class_nodeid_prefix}, but got {actual}"
