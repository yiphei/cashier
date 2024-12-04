from __future__ import annotations

from cashier.graph.new_classes import (
    ActionableMixin,
    ActionableSchemaMixin,
    AutoMixinInit,
    HasIdMixin,
)


class Node(HasIdMixin, ActionableMixin, metaclass=AutoMixinInit):
    pass


class NodeSchema(HasIdMixin, ActionableSchemaMixin, metaclass=AutoMixinInit):
    instance_cls = Node