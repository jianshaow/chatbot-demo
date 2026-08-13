from agents import RawResponsesStreamEvent

from common import openai_chat_model as model
from common.openai import get_agent, run_streamed_agent
from openai.types.responses import ResponseTextDeltaEvent


async def main():
    agent = get_agent(
        model=model, name="Assistant", instructions="You are a helpful assistant"
    )
    print("-" * 80)
    print("chat model:", model)
    result = run_streamed_agent(agent, "Write a haiku about recursion in programming.")
    print("-" * 80)
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent) and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)
    print("\n", "-" * 80, sep="")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
