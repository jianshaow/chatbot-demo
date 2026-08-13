from agents import (
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    ToolCallItem,
    ToolCallOutputItem,
)

from common import openai_chat_model as model
from common.fn_tools import calc_tools
from common.openai import get_agent, run_streamed_agent
from common.prompts import fn_call_adv_question as question
from common.prompts import fn_call_system as system_prompt
from openai.types.responses import ResponseFunctionToolCall, ResponseTextDeltaEvent


async def main():
    agent = get_agent(
        model=model,
        name="Assistant",
        instructions=system_prompt,
        tools=list(calc_tools),
    )
    print("-" * 80)
    print("chat model:", model)

    result = run_streamed_agent(agent, question)
    print("-" * 80)
    streaming_started = False
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
                streaming_started = True
                print(event.data.delta, end="", flush=True)
        elif isinstance(event, RunItemStreamEvent):
            if isinstance(event.item, ToolCallItem) and isinstance(
                event.item.raw_item, ResponseFunctionToolCall
            ):
                if streaming_started:
                    print("\n", "-" * 80, sep="")
                    streaming_started = False
                print(
                    f"{event.item.tool_name} called with args: {event.item.raw_item.arguments}"
                )
                print("-" * 80)
            elif isinstance(event.item, ToolCallOutputItem):
                if streaming_started:
                    print("\n", "-" * 80, sep="")
                    streaming_started = False
                print(f"Tool output: {event.item.output}")
                print("-" * 80)

    print("\n", "-" * 80, sep="")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
