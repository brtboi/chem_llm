"""Model backends (LangChain chat models) and finding or starting the model server."""


def __getattr__(name):
    # chem_llm.llm was one module (now llm.models); `from chem_llm.llm import
    # TransformersChatModel` keeps working without importing LangChain eagerly.
    import importlib

    # not `from . import models`: its hasattr() check would land back here
    models = importlib.import_module(f"{__name__}.models")
    try:
        return getattr(models, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
