from typing import List
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool
from common.functions import multiply, add

tools: List[BaseTool] = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=add),
]
