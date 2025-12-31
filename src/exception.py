import sys
import traceback
from typing import Optional

class CustomException(Exception):
    """
    Custom exception that wraps any original exception with detailed context.
    
    Attributes:
        original_exception: The original exception instance
        message: Formatted error message including file, line, and optionally full traceback
        include_traceback: Whether to include full traceback in the message
    """

    def __init__(self, error: Exception, include_traceback: bool = False):
        self.original_exception = error
        self.include_traceback = include_traceback

        # Build message only if we have traceback info
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is None:
            # No traceback available; just use the original error message
            self.message = f"{type(error).__name__}: {error}"
        else:
            self.message = self._format_error(exc_tb, error, include_traceback)

        super().__init__(self.message)

    def _format_error(self, tb, error: Exception, include_traceback: bool) -> str:
        """
        Format the error message with file, line, and optionally full traceback.
        """
        # Get last frame where the exception occurred
        last_tb = traceback.extract_tb(tb)[-1]
        file_name = last_tb.filename
        line_number = last_tb.lineno

        base_message = (
            f"Exception type: {type(error).__name__}\n"
            f"File: {file_name}\n"
            f"Line: {line_number}\n"
            f"Message: {error}"
        )

        if include_traceback:
            tb_str = ''.join(traceback.format_exception(type(error), error, tb))
            return f"{base_message}\nFull traceback:\n{tb_str}"

        return base_message

    def __str__(self) -> str:
        return self.message
    
if __name__ == "__main__":

    try:
        x = 1 / 0
    except Exception as e:
        raise CustomException(e, include_traceback=False)