class AccessKeys:
    
    def __init__(self) -> None:
        from dotenv import dotenv_values
        from helpers.utils import get_absolute_path
        tokens_dict = dotenv_values(get_absolute_path(".env.development"))

        self.HF_TOKEN = tokens_dict["HF_TOKEN"]

    
ACCESS_KEYS = AccessKeys()
