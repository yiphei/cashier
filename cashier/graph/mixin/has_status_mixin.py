from enum import StrEnum

class Status(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class HasStatusMixin:
    def __init__(self):
        self.status = Status.IN_PROGRESS

    def mark_as_completed(self) -> None:
        self.status = Status.COMPLETED
