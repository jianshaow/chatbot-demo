from typing import List

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool

from common.calc_func import add, multiply

calc_tools: List[BaseTool] = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=add),
]
