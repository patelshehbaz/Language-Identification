import os 
from src.exception import CustomException
import dill, os, sys

import base64

def decodesound(string, filename):
    """
    It takes a base64 encoded string and writes it to a file
    
    Args:
      string: The base64 encoded string
      filename: The name of the file you want to save the sound as.
    """
    data = base64.b64decode(string)
    with open(filename, 'wb') as f:
        f.write(data)
        f.close()

def save_object(file_path: str, obj: object) -> None:
    """
    It creates a directory if it doesn't exist, and then saves the object to the file path
    
    Args:
      file_path (str): The path to the file to save the object to.
      obj (object): The object to be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    """
    It loads an object from a file
    
    Args:
      file_path (str): str
    
    Returns:
      The object that was saved in the file.
    """
    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj

    except Exception as e:
        raise CustomException(e, sys) 