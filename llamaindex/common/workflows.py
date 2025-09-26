from typing import cast

from llama_index.core.agent.workflow import AgentStream, AgentWorkflow
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from workflows.events import Event


class ReActContext(Context):
    def __init__(self, *args, **kwargs):
        self.buffer = ""
        self.started = False
        super().__init__(*args, **kwargs)

    def write_event_to_stream(self, ev: Event | None) -> None:
        if isinstance(ev, AgentStream):
            if self.started:
                super().write_event_to_stream(ev)
            else:
                self.buffer += ev.delta
                if "Answer:" in self.buffer:
                    self.started = True
                    ev.delta = self.buffer.split("Answer: ")[-1]
                    super().write_event_to_stream(ev)

        else:
            self.buffer = ""
            self.started = False
            super().write_event_to_stream(ev)


def from_tools_or_functions(*args, **kwargs) -> AgentWorkflow:
    agent = AgentWorkflow.from_tools_or_functions(*args, **kwargs)

    def wrap_run(func):
        ctx = ReActContext(agent)

        def wrapper(*args, **kwargs):
            return func(ctx=ctx, *args, **kwargs)

        return wrapper

    llm = args[1] or kwargs.get("llm")
    if llm:
        llm = cast(LLM, llm)
        if not llm.metadata.is_function_calling_model:
            agent.run = wrap_run(agent.run)

    return agent
