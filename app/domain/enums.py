from enum import StrEnum


class Platform(StrEnum):
    IOS = "ios"
    ANDROID = "android"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
