from __future__ import annotations

from cashier.graph.auto_mixin_init import AutoMixinInit
from cashier.graph.has_id_mixin import HasIdMixin
from cashier.graph.new_classes import (
    HasActionableMixin,
    HasActionableSchemaMixin,
)


class Node(HasIdMixin, HasActionableMixin, metaclass=AutoMixinInit):
    pass


class NodeSchema(HasIdMixin, HasActionableSchemaMixin, metaclass=AutoMixinInit):
    instance_cls = Node
