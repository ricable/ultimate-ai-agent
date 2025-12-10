class UserError(Exception):
    """
    Exception raised for user-facing validation errors or other issues
    resulting from invalid user input or actions.

    This exception is intended to be caught and its message displayed to
    the user, providing helpful feedback on what went wrong. It is distinct
    from internal errors or bugs in the application.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes a new UserError exception.

        Args:
            message: The error message to display to the user. This message
                should be clear, concise, and informative, explaining the reason
                for the error.

        Returns:
            None.
        """
        super().__init__(message)

    def __str__(self) -> str:
        """
        Returns a string representation of the UserError.

        This method overrides the default __str__ method to provide a
        cleaner, user-friendly message, removing the surrounding parentheses
        and quotes that would otherwise be present in the default representation.

        Returns:
            The error message as a string.
        """
        return f"{self.args[0]}"
