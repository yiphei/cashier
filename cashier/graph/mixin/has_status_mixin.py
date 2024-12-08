from enum import StrEnum


class Status(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    TRANSITIONING = "TRANSITIONING"
    COMPLETED = "COMPLETED"


class HasStatusMixin:
    def __init__(self):
        self.status = Status.IN_PROGRESS

    def mark_as_completed(self) -> None:
        self.status = Status.COMPLETED

    def mark_as_transitioning(self) -> None:
        self.status = Status.TRANSITIONING