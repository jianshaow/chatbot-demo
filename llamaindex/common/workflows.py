from typing import cast

from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentStream, AgentWorkflow
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from workflows.events import Event

FINAL_ANSWER_PREFIX = "Answer: "


class ReActContext(Context):
    def __init__(self, *args, **kwargs):
        self.buffer = ""
        self.started = False
        self.final_answer = ""
        super().__init__(*args, **kwargs)

    def write_event_to_stream(self, ev: Event | None) -> None:
        if ev is None:
            return
        if isinstance(ev, AgentStream):
            if self.started:
                self.final_answer += ev.delta
                ev.response = self.final_answer
                super().write_event_to_stream(ev)
            else:
                self.buffer += ev.delta
                if FINAL_ANSWER_PREFIX in self.buffer:
                    self.started = True
                    ev.delta = self.buffer.split(FINAL_ANSWER_PREFIX)[-1]
                    ev.response = self.final_answer = ev.delta
                    super().write_event_to_stream(ev)

        else:
            self.buffer = ""
            self.started = False
            self.final_answer = ""
            super().write_event_to_stream(ev)


class FunctionContext(Context):
    def write_event_to_stream(self, ev: Event | None) -> None:
        if ev is None:
            return
        if isinstance(ev, AgentStream):
            if ev.response != "":
                super().write_event_to_stream(ev)
        else:
            super().write_event_to_stream(ev)


def from_tools_or_functions(*args, **kwargs) -> AgentWorkflow:
    agent = AgentWorkflow.from_tools_or_functions(*args, **kwargs)

    llm = args[1] or kwargs.get("llm") or Settings.llm
    llm = cast(LLM, llm)
    if llm.metadata.is_function_calling_model:
        ctx = FunctionContext(agent)
    else:
        ctx = ReActContext(agent)

    def wrap_run(func):
        def wrapper(*args, **kwargs):
            return func(ctx=ctx, *args, **kwargs)

        return wrapper

    agent.run = wrap_run(agent.run)
    return agent
