from __future__ import annotations

from cashier.graph.mixin.auto_mixin_init import AutoMixinInit
from cashier.graph.mixin.has_chat_mixin import HasChatMixin, HasChatSchemaMixin
from cashier.graph.mixin.has_id_mixin import HasIdMixin


class Node(HasIdMixin, HasChatMixin, metaclass=AutoMixinInit):
    pass


class NodeSchema(HasIdMixin, HasChatSchemaMixin, metaclass=AutoMixinInit):
    instance_cls = Node
