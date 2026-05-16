class AppError(Exception):
    status_code: int = 500

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class AuthError(AppError):
    status_code = 401


class NotFoundError(AppError):
    status_code = 404


class ValidationError(AppError):
    status_code = 422


class UpstreamError(AppError):
    status_code = 502
