from agents import RawResponsesStreamEvent

from common import openai_chat_model as model
from common.openai import get_agent, run_streamed_agent
from common.prompts import chat_question as question
from common.prompts import chat_system as system_prompt
from openai.types.responses import ResponseTextDeltaEvent


async def main():
    agent = get_agent(model=model, name="Assistant", instructions=system_prompt)
    print("-" * 80)
    print("chat model:", model)
    result = run_streamed_agent(agent, question)
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
