from agents import function_tool

from common.functions import add, multiply
from openai.types.chat import ChatCompletionFunctionToolParam

tools: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "multiply(a: int, b: int) -> int",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                    },
                    "b": {
                        "type": "integer",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
                "description": "add(a: int, b: int) -> int",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                    },
                    "b": {
                        "type": "integer",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
]

calc_tools = [function_tool(add), function_tool(multiply)]
