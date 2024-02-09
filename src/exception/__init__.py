import os

def error_message_detail(error, error_detail):
    """
    It takes an error and an error detail and returns a string with the file name, line number, and
    error message
    
    Args:
      error: The error message that was raised.
      error_detail: This is the error detail object that is passed to the error handler.
    
    Returns:
      The error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


# It's a custom exception class that takes an error message and error detail as arguments and returns 
# a formatted error message
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message