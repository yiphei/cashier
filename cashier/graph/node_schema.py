from __future__ import annotations

from cashier.graph.new_classes import (
    AutoMixinInit,
    HasActionableMixin,
    HasActionableSchemaMixin,
    HasIdMixin,
)


class Node(HasIdMixin, HasActionableMixin, metaclass=AutoMixinInit):
    pass


class NodeSchema(HasIdMixin, HasActionableSchemaMixin, metaclass=AutoMixinInit):
    instance_cls = Node
