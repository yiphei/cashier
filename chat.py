import argparse
import copy
import json
import os
import tempfile
from collections import defaultdict, deque
from distutils.util import strtobool
from types import GeneratorType
from typing import Dict, List, Set

from colorama import Fore, Style
from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream
from pydantic import BaseModel, ConfigDict, Field

from audio import get_audio_input, save_audio_to_wav
from chain import (
    BACKGROUND,
    EDGE_SCHEMA_ID_TO_EDGE_SCHEMA,
    FROM_NODE_SCHEMA_ID_TO_EDGE_SCHEMA,
    NODE_SCHEMA_ID_TO_NODE_SCHEMA,
    Direction,
    Edge,
    EdgeSchema,
    FwdSkipType,
    Node,
    take_order_node_schema,
)
from db_functions import create_db_client
from function_call_context import FunctionCallContext, InexistentFunctionError
from gui import remove_previous_line
from logger import logger
from model import CustomJSONEncoder, Model, TurnContainer
from model_tool_decorator import ToolRegistry
from model_util import ModelProvider

# Load environment variables from .env file
load_dotenv()


def get_text_from_speech(audio_data, oai_client):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
        # Save the audio data as a WAV file
        save_audio_to_wav(audio_data, temp_wav_file.name)

        # Use OpenAI's API to transcribe the saved WAV file
        transcription = oai_client.audio.transcriptions.create(
            model="whisper-1",
            language="en",
            file=open(temp_wav_file.name, "rb"),  # Open the saved WAV file for reading
        )
    return transcription.text


def get_speech_from_text(text_iterator, elabs_client):
    audio = elabs_client.generate(
        voice=Voice(
            voice_id="cgSgspJ2msm6clMCkdW9",
            settings=VoiceSettings(
                stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
            ),
        ),
        text=text_iterator,
        model="eleven_multilingual_v2",
        stream=True,
        optimize_streaming_latency=3,
    )
    stream(audio)


def get_user_input(use_audio_input, openai_client):
    if use_audio_input:
        audio_input = get_audio_input()
        text_input = get_text_from_speech(audio_input, openai_client)
    else:
        text_input = input("You: ")
        remove_previous_line()

    return text_input


class MessageDisplay:
    API_ROLE_TO_PREFIX = {
        "system": "System",
        "user": "You",
        "assistant": "Assistant",
    }

    API_ROLE_TO_COLOR = {
        "system": Fore.GREEN,
        "user": Fore.WHITE,
        "assistant": Fore.MAGENTA,
    }

    @classmethod
    def display_assistant_message(cls, message_or_stream):
        if isinstance(message_or_stream, GeneratorType):
            cls.print_msg(role="assistant", msg=None, end="")
            full_msg = ""
            for msg_chunk in message_or_stream:
                cls.print_msg(
                    role="assistant", msg=msg_chunk, add_role_prefix=False, end=""
                )
                full_msg += msg_chunk

            print("\n\n")
        else:
            cls.print_msg("assistant", message_or_stream)

    @classmethod
    def get_role_prefix(cls, role):
        return f"{Style.BRIGHT}{cls.API_ROLE_TO_PREFIX[role]}: {Style.NORMAL}"

    @classmethod
    def print_msg(cls, role, msg, add_role_prefix=True, end="\n\n"):
        formatted_msg = f"{cls.API_ROLE_TO_COLOR[role]}"
        if add_role_prefix:
            formatted_msg += f"{cls.get_role_prefix(role)}"
        if msg is not None:
            formatted_msg += f"{msg}"

        formatted_msg += f"{Style.RESET_ALL}"
        print(
            formatted_msg,
            end=end,
        )


def should_skip_node_schema(model, TM, current_node_schema, all_node_schemas):
    if len(all_node_schemas) == 1:
        return None

    model_name = "claude-3.5"
    model_provider = Model.get_model_provider(model_name)
    conversational_msgs = copy.deepcopy(
        TM.model_provider_to_message_manager[
            model_provider
        ].get_conversation_msgs_since_last_node()
    )
    prompt = (
        "You are an AI-agent orchestration engine and your job is to evaluate the current AI agent's performance. "
        "The AI agent's background is:\n"
        "<background>\n"
        f"{BACKGROUND}\n"
        "</background>\n\n"
        "The AI agent is defined by 3 attributes: instructions, state, and tools (i.e. functions).\n\n"
        "The instructions describe what the agent's conversation is supposed to be about and what they are expected to do.\n"
        "<instructions>\n"
        f"{current_node_schema.node_prompt}\n"
        "</instructions>\n\n"
        "The state keeps track of important data during the conversation.\n"
        "<state>\n"
        f"{current_node_schema.state_pydantic_model.model_json_schema()}\n"
        "</state>\n\n"
        "The tools represent explicit actions that the agent can perform.\n"
        "<tools>\n"
        f"{json.dumps(ToolRegistry.get_tool_defs_from_names(current_node_schema.tool_fn_names, model_provider, current_node_schema.tool_registry))}\n"
        "</tools>\n\n"
        "Given a conversation between a customer and the current AI agent, determine if the"
        " conversation, especially given the last customer message, can continue to be fully handled by the current AI agent's <instructions>, <state>, or <tools> according to the guidelines defined in <guidelines>. Return true if"
        " 100% certain, and return false if otherwise, meaning that we should at least explore letting another AI agent take over.\n\n"
        "<guidelines>\n"
        "<state_guidelines>\n"
        "- Among the tools provided, there are functions for getting and updating the state defined in <state>. "
        "For state updates, the agent will have field specific update functions, whose names are `update_state_<field>` and where <field> is a state field.\n"
        "- The agent must update the state whenever applicable and as soon as possible. They cannot proceed to the next stage of the conversation without updating the state\n"
        "- Only the agent can update the state, so there is no need to udpate the state to the same value that had already been updated to in the past.\n"
        + "</state_guidelines>\n"
        "<tools_guidelines>\n"
        "- Minimize reliance on external knowledge. Always retrieve information from the system prompts and available tools. "
        "If they dont provide the information needed, the agent must say they do not know.\n"
        "- the agent must AVOID stating/mentioning that they can/will perform an action if there are no tools (including state updates) associated with that action.\n"
        "- if the agent needs to perform an action, they can only state to the customer that they performed it after the associated tool (including state update) calls have been successfull.\n"
        "</tools_guidelines>\n"
        "<general_guidelines>\n"
        "- the agent needs to think step-by-step before responding.\n"
        "- the agent must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
        + "</general_guidelines>\n"
        "</guidelines>\n\n"
        "<last_customer_message>\n"
        f"{conversational_msgs[-1]['content']}\n"
        "</last_customer_message>\n\n"
    )
    if model_provider == ModelProvider.ANTHROPIC:
        conversational_msgs.append({"role": "user", "content": prompt})
    elif model_provider == ModelProvider.OPENAI:
        conversational_msgs.append({"role": "system", "content": prompt})

    class Response1(BaseModel):
        output: bool

    chat_completion = model.chat(
        model_name=model_name,
        message_dicts=conversational_msgs,
        response_format=Response1,
        logprobs=True,
        temperature=0,
    )
    is_on_topic = chat_completion.get_message_prop("output")
    if model_provider == ModelProvider.OPENAI:
        prob = chat_completion.get_prob(-2)
        logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
    else:
        logger.debug(f"IS_ON_TOPIC: {is_on_topic}")

    if is_on_topic:
        return None

    conversational_msgs.pop()
    prompt = (
        "You are an AI-agent orchestration engine and your job is to select the best AI agent. "
        "Each AI agent is defined by 3 attributes: instructions, state, and tools (i.e. functions). "
        "The instructions <instructions> describe what the agent's conversation is supposed to be about and what they are expected to do. "
        "The state <state> keeps track of important data during the conversation. "
        "The tools <tools> represent explicit actions that the agent can perform.\n\n"
    )
    for node_schema in all_node_schemas:
        prompt += (
            f"<agent id={node_schema.id}>\n"
            "<instructions>\n"
            f"{node_schema.node_prompt}\n"
            "</instructions>\n\n"
            "<state>\n"
            f"{node_schema.state_pydantic_model.model_json_schema()}\n"
            "</state>\n\n"
            "<tools>\n"
            f"{json.dumps(ToolRegistry.get_tool_defs_from_names(node_schema.tool_fn_names, model_provider, node_schema.tool_registry))}\n"
            "</tools>\n"
            "</agent>\n\n"
        )

    prompt += (
        "All agents share the following background:\n"
        "<background>\n"
        f"{BACKGROUND}\n"
        "</background>\n\n"
        "Given a conversation with a customer and the list above of AI agents with their attributes, "
        "determine which AI agent can best continue the conversation, especially given last customer message, in accordance with the universal guidelines defined in <guidelines>. "
        "Respond by returning the AI agent ID.\n\n"
        "<guidelines>\n"
        "<state_guidelines>\n"
        "- Among the tools provided, there are functions for getting and updating the state defined in <state>. "
        "For state updates, the agent will have field specific update functions, whose names are `update_state_<field>` and where <field> is a state field.\n"
        "- The agent must update the state whenever applicable and as soon as possible. They cannot proceed to the next stage of the conversation without updating the state\n"
        "- Only the agent can update the state, so there is no need to udpate the state to the same value that had already been updated to in the past.\n"
        + "</state_guidelines>\n"
        "<tools_guidelines>\n"
        "- Minimize reliance on external knowledge. Always retrieve information from the system prompts and available tools. "
        "If they dont provide the information needed, the agent must say they do not know.\n"
        "- the agent must AVOID stating/mentioning that they can/will perform an action if there are no tools (including state updates) associated with that action.\n"
        "- if the agent needs to perform an action, they can only state to the customer that they performed it after the associated tool (including state update) calls have been successfull.\n"
        "</tools_guidelines>\n"
        "<general_guidelines>\n"
        "- the agent needs to think step-by-step before responding.\n"
        "- the agent must decline to do anything that is not explicitly covered by <instructions> and <guidelines>.\n"
        + "</general_guidelines>\n"
        "</guidelines>\n\n"
        "<last_customer_message>\n"
        f"{conversational_msgs[-1]['content']}\n"
        "</last_customer_message>\n\n"
    )

    class Response2(BaseModel):
        agent_id: int

    if model_provider == ModelProvider.ANTHROPIC:
        conversational_msgs.append({"role": "user", "content": prompt})
    elif model_provider == ModelProvider.OPENAI:
        conversational_msgs.append({"role": "system", "content": prompt})

    chat_completion = model.chat(
        model_name=model_name,
        message_dicts=conversational_msgs,
        response_format=Response2,
        logprobs=True,
        temperature=0,
    )

    agent_id = chat_completion.get_message_prop("agent_id")
    if model_provider == ModelProvider.OPENAI:
        prob = chat_completion.get_prob(-2)
        logger.debug(f"AGENT_ID: {agent_id} with {prob}")
    else:
        logger.debug(f"AGENT_ID: {agent_id}")

    return agent_id if agent_id != current_node_schema.id else None


def handle_skip(model, TC, CT):
    fwd_skip_edge_schemas = CT.compute_fwd_skip_edge_schemas()
    bwd_skip_edge_schemas = CT.bwd_skip_edge_schemas

    all_node_schemas = [CT.curr_node.schema]
    all_node_schemas += [edge.to_node_schema for edge in fwd_skip_edge_schemas]
    all_node_schemas += [edge.from_node_schema for edge in bwd_skip_edge_schemas]

    node_schema_id = should_skip_node_schema(
        model, TC, CT.curr_node.schema, all_node_schemas
    )

    if node_schema_id is not None:
        for edge_schema in fwd_skip_edge_schemas:
            if edge_schema.to_node_schema.id == node_schema_id:
                return edge_schema, NODE_SCHEMA_ID_TO_NODE_SCHEMA[node_schema_id]

        for edge_schema in bwd_skip_edge_schemas:
            if edge_schema.from_node_schema.id == node_schema_id:
                return edge_schema, NODE_SCHEMA_ID_TO_NODE_SCHEMA[node_schema_id]

    return None, None


class ChatContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    remove_prev_tool_calls: bool
    curr_node: Node = None
    need_user_input: bool = True
    edge_schema_id_to_edges: Dict[str, List[Edge]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    from_node_id_to_edge_schema_id: Dict[str, str] = Field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    next_edge_schemas: Set[EdgeSchema] = Field(default_factory=set)
    bwd_skip_edge_schemas: Set[EdgeSchema] = Field(default_factory=set)

    def add_edge(self, from_node, to_node, edge_schema_id):
        self.edge_schema_id_to_edges[edge_schema_id].append(Edge(from_node, to_node))
        self.from_node_id_to_edge_schema_id[from_node.id] = edge_schema_id

    def get_edge_by_edge_schema_id(self, edge_schema_id, idx=-1):
        return (
            self.edge_schema_id_to_edges[edge_schema_id][idx]
            if len(self.edge_schema_id_to_edges[edge_schema_id]) >= abs(idx)
            else None
        )

    def get_prev_node(self, edge_schema, direction):
        if edge_schema and self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
            from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
            return to_node if direction == Direction.FWD else from_node
        else:
            return None

    def init_node_core(
        self,
        node_schema,
        edge_schema,
        TC,
        input,
        last_msg,
        prev_node,
        direction,
        is_skip=False,
    ):
        logger.debug(
            f"[NODE_SCHEMA] Initializing node with {Style.BRIGHT}node_schema_id: {node_schema.id}{Style.NORMAL}"
        )
        new_node = node_schema.create_node(
            input, last_msg, prev_node, edge_schema, direction
        )

        TC.add_node_turn(
            new_node,
            remove_prev_tool_calls=self.remove_prev_tool_calls,
            is_skip=is_skip,
        )
        MessageDisplay.print_msg("system", new_node.prompt)

        if node_schema.first_turn and prev_node is None:
            TC.add_assistant_direct_turn(node_schema.first_turn)
            MessageDisplay.print_msg("assistant", node_schema.first_turn.msg_content)

        if edge_schema:
            if direction == Direction.FWD:
                immediate_from_node = self.curr_node
                if edge_schema.from_node_schema != self.curr_node.schema:
                    from_node, _ = self.get_edge_by_edge_schema_id(edge_schema.id)
                    immediate_from_node = from_node
                    while from_node.schema != self.curr_node.schema:
                        prev_edge_schema = from_node.in_edge_schema
                        from_node, to_node = self.get_edge_by_edge_schema_id(
                            prev_edge_schema.id
                        )

                    self.add_edge(self.curr_node, to_node, prev_edge_schema.id)

                self.add_edge(immediate_from_node, new_node, edge_schema.id)
            elif direction == Direction.BWD and new_node.in_edge_schema:
                from_node, _ = self.get_edge_by_edge_schema_id(
                    new_node.in_edge_schema.id
                )
                self.add_edge(from_node, new_node, new_node.in_edge_schema.id)

                _, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
                self.add_edge(new_node, to_node, edge_schema.id)

        self.curr_node = new_node
        self.next_edge_schemas = set(
            FROM_NODE_SCHEMA_ID_TO_EDGE_SCHEMA.get(new_node.schema.id, [])
        )
        self.compute_bwd_skip_edge_schemas()

    def init_next_node(self, node_schema, edge_schema, TC, input=None):
        if self.curr_node:
            self.curr_node.mark_as_completed()

        if input is None and edge_schema:
            input = edge_schema.new_input_from_state_fn(self.curr_node.state)

        if edge_schema:
            edge_schema, input = self.compute_next_edge_schema(edge_schema, input)
            node_schema = edge_schema.to_node_schema

        direction = Direction.FWD
        prev_node = self.get_prev_node(edge_schema, direction)

        mm = TC.model_provider_to_message_manager[ModelProvider.OPENAI]
        last_msg = mm.get_user_message()
        if last_msg:
            last_msg = last_msg["content"]

        self.init_node_core(
            node_schema, edge_schema, TC, input, last_msg, prev_node, direction, False
        )

    def init_skip_node(
        self,
        node_schema,
        edge_schema,
        TC,
    ):
        direction = Direction.FWD
        if edge_schema and edge_schema.from_node_schema == node_schema:
            direction = Direction.BWD

        if direction == Direction.BWD:
            self.bwd_skip_edge_schemas.clear()

        prev_node = self.get_prev_node(edge_schema, direction)
        input = prev_node.input

        mm = TC.model_provider_to_message_manager[ModelProvider.OPENAI]
        last_msg = mm.get_asst_message()
        if last_msg:
            last_msg = last_msg["content"]

        self.init_node_core(
            node_schema, edge_schema, TC, input, last_msg, prev_node, direction, True
        )

    def compute_bwd_skip_edge_schemas(self):
        from_node = self.curr_node
        while from_node.in_edge_schema is not None:
            if from_node.in_edge_schema in self.bwd_skip_edge_schemas:
                return
            self.bwd_skip_edge_schemas.add(from_node.in_edge_schema)
            new_from_node, to_node = self.get_edge_by_edge_schema_id(
                from_node.in_edge_schema.id
            )
            assert from_node == to_node
            from_node = new_from_node

    def compute_fwd_skip_edge_schemas(self):
        fwd_jump_edge_schemas = set()
        edge_schemas = deque(self.next_edge_schemas)
        while edge_schemas:
            edge_schema = edge_schemas.popleft()
            if self.get_edge_by_edge_schema_id(edge_schema.id) is not None:
                from_node, to_node = self.get_edge_by_edge_schema_id(edge_schema.id)
                if from_node.schema == self.curr_node.schema:
                    from_node = self.curr_node

                if edge_schema.can_skip(
                    from_node,
                    to_node,
                    self.is_prev_from_node_completed(
                        edge_schema, from_node == self.curr_node
                    ),
                )[0]:
                    fwd_jump_edge_schemas.add(edge_schema)
                    more_edges = FROM_NODE_SCHEMA_ID_TO_EDGE_SCHEMA.get(
                        to_node.schema.id, []
                    )
                    edge_schemas.extend(more_edges)

        return fwd_jump_edge_schemas

    def is_prev_from_node_completed(self, edge_schema, is_start_node):
        idx = -1 if is_start_node else -2
        edge = self.get_edge_by_edge_schema_id(edge_schema.id, idx)
        return edge[0].status == Node.Status.COMPLETED if edge else False

    def compute_next_edge_schema(self, start_edge_schema, start_input):
        next_edge_schema = start_edge_schema
        edge_schema = start_edge_schema
        input = start_input
        while self.get_edge_by_edge_schema_id(next_edge_schema.id) is not None:
            from_node, to_node = self.get_edge_by_edge_schema_id(next_edge_schema.id)
            if from_node.schema == self.curr_node.schema:
                from_node = self.curr_node

            next_next_edge_schema_id = self.from_node_id_to_edge_schema_id[to_node.id]
            next_next_edge_schema = (
                EDGE_SCHEMA_ID_TO_EDGE_SCHEMA[next_next_edge_schema_id]
                if next_next_edge_schema_id
                else None
            )
            can_skip, skip_type = next_edge_schema.can_skip(
                from_node,
                to_node,
                self.is_prev_from_node_completed(
                    next_edge_schema, from_node == self.curr_node
                ),
            )

            if can_skip:
                edge_schema = next_edge_schema
                if next_next_edge_schema:
                    next_edge_schema = next_next_edge_schema
                else:
                    input = to_node.input
                    break
            else:
                if skip_type == FwdSkipType.SKIP_IF_INPUT_UNCHANGED:
                    if from_node.status != Node.Status.COMPLETED:
                        input = from_node.input
                    else:
                        edge_schema = next_edge_schema
                        if from_node != self.curr_node:
                            input = edge_schema.new_input_from_state_fn(from_node.state)
                else:
                    if from_node != self.curr_node:
                        input = from_node.input
                break

        return edge_schema, input


def run_chat(args, model, elevenlabs_client):
    TC = TurnContainer()
    CT = ChatContext(remove_prev_tool_calls=args.remove_prev_tool_calls)
    CT.init_next_node(take_order_node_schema, None, TC, None)

    while True:
        force_tool_choice = None

        if CT.need_user_input:
            # Read user input from stdin
            text_input = get_user_input(args.audio_input, model.oai_client)
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break
            MessageDisplay.print_msg("user", text_input)
            TC.add_user_turn(text_input)
            skip_edge_schema, skip_node_schema = handle_skip(model, TC, CT)
            if skip_edge_schema is not None:
                CT.init_skip_node(
                    skip_node_schema,
                    skip_edge_schema,
                    TC,
                )
                force_tool_choice = "get_state"
            CT.curr_node.update_first_user_message()

        chat_completion = model.chat(
            model_name=args.model,
            turn_container=TC,
            tool_names_or_tool_defs=CT.curr_node.schema.tool_fn_names,
            stream=args.stream,
            extra_tool_registry=CT.curr_node.schema.tool_registry,
            force_tool_choice=force_tool_choice,
        )
        message = chat_completion.get_or_stream_message()
        if message is not None:
            if args.audio_output:
                get_speech_from_text(message, elevenlabs_client)
            else:
                MessageDisplay.display_assistant_message(message)
            CT.need_user_input = True

        fn_id_to_output = {}
        new_edge_schema = None
        for function_call in chat_completion.get_or_stream_fn_calls():
            function_args = function_call.function_args
            logger.debug(
                f"[FUNCTION_CALL] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
            )
            with FunctionCallContext() as fn_call_context:
                if function_call.function_name not in CT.curr_node.schema.tool_fn_names:
                    raise InexistentFunctionError(function_call.function_name)

                if function_call.function_name.startswith("get_state"):
                    fn_output = getattr(CT.curr_node, function_call.function_name)(
                        **function_args
                    )
                elif function_call.function_name.startswith("update_state"):
                    fn_output = CT.curr_node.update_state(**function_args)
                else:
                    fn = ToolRegistry.GLOBAL_FN_NAME_TO_FN[function_call.function_name]
                    fn_output = fn(**function_args)

            if fn_call_context.has_exception():
                logger.debug(
                    f"[FUNCTION_EXCEPTION] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with exception:\n{str(fn_call_context.exception)}"
                )
                fn_id_to_output[function_call.tool_call_id] = fn_call_context.exception
            else:
                logger.debug(
                    f"[FUNCTION_RETURN] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with output:\n{json.dumps(fn_output, cls=CustomJSONEncoder, indent=4)}"
                )
                fn_id_to_output[function_call.tool_call_id] = fn_output

            CT.need_user_input = False

            if (
                not fn_call_context.has_exception()
                and function_call.function_name.startswith("update_state")
            ):
                for edge_schema in CT.next_edge_schemas:
                    if edge_schema.check_state_condition(CT.curr_node.state):
                        new_edge_schema = edge_schema
                        break

        model_provider = Model.get_model_provider(args.model)
        TC.add_assistant_turn(
            chat_completion.msg_content,
            model_provider,
            chat_completion.fn_calls,
            fn_id_to_output,
        )

        if new_edge_schema:
            new_node_schema = new_edge_schema.to_node_schema
            CT.init_next_node(
                new_node_schema,
                new_edge_schema,
                TC,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_input",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    parser.add_argument(
        "--audio_output",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--output_system_prompt",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--enable_logging",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--stream",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--remove_prev_tool_calls",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    args = parser.parse_args()

    model = Model()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_db_client()

    if not args.enable_logging:
        logger.disabled = True
    run_chat(args, model, elevenlabs_client)
