from lib import *


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()
        gc.collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        gc.collect()