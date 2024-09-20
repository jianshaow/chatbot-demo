from llama_index.core.tools import FunctionTool
from common.functions import multiply, add

tools = [FunctionTool.from_defaults(fn=multiply), FunctionTool.from_defaults(fn=add)]
