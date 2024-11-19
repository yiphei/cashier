import copy
from bisect import bisect_left
from collections import defaultdict
from enum import StrEnum
from typing import Any, Dict, Iterable, List, Set, SupportsIndex, overload, Optional, Union

from cashier.model.model_util import ModelProvider


class MessageList(list):
    class ItemType(StrEnum):
        USER = "USER"
        ASSISTANT = "ASSISTANT"
        TOOL_CALL = "TOOL_CALL"
        TOOL_OUTPUT = "TOOL_OUTPUT"
        TOOL_OUTPUT_SCHEMA = "TOOL_OUTPUT_SCHEMA"
        NODE = "NODE"

    item_type_to_uri_prefix = {
        ItemType.USER: "usr_",
        ItemType.TOOL_OUTPUT: "tout_",
        ItemType.NODE: "node_",
        ItemType.ASSISTANT: "asst_",
    }

    def __init__(self, *args: Any, model_provider: ModelProvider):
        super().__init__(*args)
        self.uri_to_list_idx: Dict[str, int] = {}
        self.list_idx_to_uris: Dict[int, Set[str]] = defaultdict(set)
        self.list_idxs: List[int] = []
        self.list_idx_to_track_idx: Dict[int, int] = {}

        self.model_provider = model_provider
        self.item_type_to_uris: Dict[MessageList.ItemType, List[str]] = defaultdict(
            list
        )
        self.uri_to_item_type: Dict[str, MessageList.ItemType] = {}
        self.item_type_to_count = {k: 0 for k in self.item_type_to_uri_prefix.keys()}

    def get_tool_id_from_tool_output_uri(self, uri: str) -> str:
        return uri[
            len(self.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT]) :
        ]

    @classmethod
    def get_tool_output_uri_from_tool_id(cls, tool_id: str) -> str:
        return cls.item_type_to_uri_prefix[MessageList.ItemType.TOOL_OUTPUT] + tool_id

    def pop_track_idx_ant(self, uri: str) -> None:
        track_idx = self.get_track_idx_from_uri(uri)
        item_type = self.uri_to_item_type[uri]
        message = self[track_idx]
        new_contents = []
        for content in message["content"]:
            if (
                item_type == MessageList.ItemType.TOOL_CALL
                and content["type"] == "tool_use"
                and content["id"] == uri
            ):
                continue
            elif (
                item_type == MessageList.ItemType.TOOL_OUTPUT
                and content["type"] == "tool_result"
                and content["tool_use_id"] == self.get_tool_id_from_tool_output_uri(uri)
            ):
                continue
            new_contents.append(content)

        if new_contents:
            if item_type == MessageList.ItemType.TOOL_CALL:
                if len(new_contents) == 1 and new_contents[0]["type"] == "text":
                    new_message = {
                        "role": "assistant",
                        "content": new_contents[0]["text"],
                    }
                else:
                    new_message = {"role": "assistant", "content": new_contents}
            elif item_type == MessageList.ItemType.TOOL_OUTPUT:
                new_message = {"role": "user", "content": new_contents}

            self[track_idx] = new_message
            self.pop_track_idx(uri, shift_idxs=False)
        else:
            self._remove_by_uri(uri, True)

    def track_idx(
        self,
        item_type: ItemType,
        list_idx: Optional[int] = None,
        uri: Optional[str] = None,
        is_insert: bool = False,
    ) -> None:
        if uri is None:
            self.item_type_to_count[item_type] += 1
            uri = self.item_type_to_uri_prefix[item_type] + str(
                self.item_type_to_count[item_type]
            )
        if list_idx is None:
            list_idx = len(self) - 1

        if uri in self.uri_to_list_idx:
            raise ValueError()

        self.uri_to_list_idx[uri] = list_idx
        self.item_type_to_uris[item_type].append(uri)
        self.uri_to_item_type[uri] = item_type
        if list_idx not in self.list_idxs or is_insert:
            if (self.list_idxs and self.list_idxs[-1] < list_idx) or not self.list_idxs:
                self.list_idxs.append(list_idx)
                self.list_idx_to_track_idx[list_idx] = len(self.list_idxs) - 1
            else:
                insert_idx = bisect_left(self.list_idxs, list_idx)

                self.list_idxs.insert(insert_idx, list_idx)
                self.shift_track_idxs(insert_idx + 1, 1)
                self.list_idx_to_track_idx[list_idx] = insert_idx

        self.list_idx_to_uris[list_idx].add(uri)

    def track_idxs(
        self,
        item_type: ItemType,
        start_list_idx: int,
        end_list_idx: Optional[int] = None,
        uris: Optional[List[Optional[str]]] = None,
    ) -> None:
        if end_list_idx is None:
            end_list_idx = len(self) - 1
        if uris is None:
            range_idx = end_list_idx - start_list_idx + 1
            uris = [None] * range_idx

        for i, uri in zip(range(start_list_idx, end_list_idx + 1), uris):
            self.track_idx(item_type, i, uri)

    def get_track_idx_from_uri(self, uri: str) -> int:
        return self.uri_to_list_idx[uri]

    def get_track_idx_for_item_type(
        self, item_type: ItemType, idx: int = -1
    ) -> Optional[int]:
        order_validation = abs(idx) if idx < 0 else idx + 1
        target_uri = (
            self.item_type_to_uris[item_type][idx]
            if self.item_type_to_uris[item_type]
            and order_validation <= len(self.item_type_to_uris[item_type])
            else None
        )
        return self.uri_to_list_idx[target_uri] if target_uri else None

    def get_item_type_by_idx(self, item_type: ItemType, idx: int) -> Any:
        track_idx = self.get_track_idx_for_item_type(item_type, idx)
        if track_idx:
            return self[track_idx]
        else:
            return None

    def shift_track_idxs(self, start_track_idx: int, shift_direction: int) -> None:
        for i in range(start_track_idx, len(self.list_idxs)):
            curr_list_idx = self.list_idxs[i]
            self.list_idx_to_track_idx.pop(curr_list_idx)
            curr_uris = self.list_idx_to_uris[curr_list_idx]

            self.list_idxs[i] += shift_direction
            self.list_idx_to_track_idx[self.list_idxs[i]] = i

            for uri in curr_uris:
                self.uri_to_list_idx[uri] = self.list_idxs[i]
            self.list_idx_to_uris.pop(curr_list_idx)
            self.list_idx_to_uris[self.list_idxs[i]] = curr_uris

    def pop_track_idx(self, uri: str, shift_idxs: bool = True) -> Optional[int]:
        popped_list_idx = self.uri_to_list_idx.pop(uri)
        all_uris = self.list_idx_to_uris[popped_list_idx]

        item_type = self.uri_to_item_type.pop(uri)
        self.item_type_to_uris[item_type].remove(uri)

        all_uris.remove(uri)
        if not all_uris:
            popped_track_idx = self.list_idx_to_track_idx.pop(popped_list_idx)
            self.list_idx_to_uris.pop(popped_list_idx)
            del self.list_idxs[popped_track_idx]

            if shift_idxs:
                self.shift_track_idxs(popped_track_idx, -1)
            else:
                for i in range(popped_track_idx, len(self.list_idxs)):
                    curr_list_idx = self.list_idxs[i]
                    self.list_idx_to_track_idx.pop(curr_list_idx)
                    self.list_idx_to_track_idx[curr_list_idx] = i

            return popped_list_idx
        else:
            return None

    def append(
        self, item: Any, item_type: Optional[ItemType] = None, uri: Optional[str] = None
    ) -> None:
        super().append(item)
        if item_type is not None:
            self.track_idx(item_type, uri=uri)

    @overload
    def insert(  # noqa: E704
        self, __index: SupportsIndex, __object: Any, /  # noqa: W504
    ) -> None: ...

    @overload
    def insert(  # noqa: E704
        self,
        idx: int,
        item: Any,
        item_type: Optional[ItemType] = None,
        uri: Optional[str] = None,
    ) -> None: ...

    def insert(
        self,
        idx: SupportsIndex,
        item: Any,
        item_type: Optional[ItemType] = None,
        uri: Optional[str] = None,
    ) -> None:
        super().insert(idx, item)
        assert isinstance(idx, int)
        if item_type is not None:
            self.track_idx(item_type, idx, uri, is_insert=True)

    @overload
    def extend(self, __iterable: Iterable[Any], /) -> None: ...  # noqa: E704

    @overload
    def extend(  # noqa: E704
        self, items: List[Any], item_type: Optional[ItemType] = None
    ) -> None: ...

    def extend(
        self,
        items: Union[List[Any], Iterable[Any]],
        item_type: Optional[ItemType] = None,
    ) -> None:
        curr_len = len(self) - 1
        super().extend(items)
        if items and item_type is not None:
            self.track_idxs(item_type, curr_len + 1)

    def _remove_by_uri(self, uri: str, raise_on_unpopped_idx: bool = False) -> None:
        popped_idx = self.pop_track_idx(uri)
        if popped_idx is not None:
            del self[popped_idx]
        else:
            if raise_on_unpopped_idx:
                raise ValueError

    def remove_by_uri(self, uri: str, raise_if_not_found: bool = True) -> None:
        if uri not in self.uri_to_item_type:
            if raise_if_not_found:
                raise ValueError()
            return

        item_type = self.uri_to_item_type[uri]
        if self.model_provider != ModelProvider.ANTHROPIC or (
            self.model_provider == ModelProvider.ANTHROPIC
            and not (
                item_type == MessageList.ItemType.TOOL_CALL
                or item_type == MessageList.ItemType.TOOL_OUTPUT
            )
        ):
            self._remove_by_uri(uri)
        else:
            self.pop_track_idx_ant(uri)

    def clear(
        self, item_type_or_types: Optional[Union[ItemType, List[ItemType]]] = None
    ) -> None:
        if item_type_or_types is None:
            super().clear()
        else:
            if not isinstance(item_type_or_types, list):
                item_type_or_types = [item_type_or_types]
            for item_type in item_type_or_types:
                uris = copy.copy(self.item_type_to_uris[item_type])
                for uri in uris:
                    self.remove_by_uri(uri)

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> Any: ...  # noqa: E704

    @overload
    def __getitem__(self, index: slice, /) -> "MessageList": ...  # noqa: E704

    def __getitem__(self, index: Union[SupportsIndex, slice]) -> Any:
        if isinstance(index, slice):
            if index.step:
                raise ValueError()

            ml = MessageList(
                super().__getitem__(index), model_provider=self.model_provider
            )
            start_idx = index.start or 0
            if start_idx < 0:
                start_idx = len(self) + start_idx
            end_idx = index.stop or len(self)
            if end_idx < 0:
                end_idx = len(self) + end_idx
            ml.item_type_to_count = copy.deepcopy(self.item_type_to_count)
            for list_idx in self.list_idxs:
                if start_idx <= list_idx < end_idx:
                    for uri in self.list_idx_to_uris[list_idx]:
                        item_type = self.uri_to_item_type[uri]
                        ml.track_idx(item_type, list_idx - start_idx, uri)

            return ml
        else:
            return super().__getitem__(index)
