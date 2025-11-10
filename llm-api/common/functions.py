from agents import function_tool


@function_tool
def multiply(a: int, b: int) -> int:
    return a * b


@function_tool
def add(a: int, b: int) -> int:
    return a + b


fns = {
    "multiply": multiply,
    "add": add,
}
