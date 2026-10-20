import os
from machine_list import MACHINE_LISTNAME
from retrevier import EdgeHybridSearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_google_genai.llms import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
import dotenv
dotenv.load_dotenv()
# from langgraph.types import Command


EXPERT_PERSONALITY = """
You are a consultant with decade of experience in industrial maintenance for machines used in factory.
You are only allowed to provide  response to user query from the provided context.
You should clarify the every information with the user to ensure that the right answers are provided.
Do not give answers outside the context, and let it be know when the questions can not be answered.
"""


# === Definition of State, Context, Response
class ExpertState(AgentState):
    pass


class ExpertContext(BaseModel):
    pass


class ExpertResponse(BaseModel):
    """Expert reponse with simplicity"""
    speech: str = Field(description="The expert textual reponse")
    machine: str = Field(
        description="The current machine currently been examined")
    suggestions: list[str] = Field(
        description="Suggested question the user might want to ask?")
    confidence: str = Field(
        description="The degreee of relevance of the provided response?")


# === Middleware ===
# define system prompts
@dynamic_prompt
def define_system_prompt(request: ModelRequest) -> SystemMessage:
    return SystemMessage(EXPERT_PERSONALITY)


# ==== Tools ====
ExpertToolRuntime = ToolRuntime[ExpertContext, ExpertState]


class ExpertAgent:
    def __init__(self, shard_path: str):
        # Long-Term Storage
        self.store = InMemoryStore()

        # Short-Term Storage
        self.checkpointer = InMemorySaver()

        # Reasoning Model Initialization
        if os.getenv("USE_REMOTE_MODEL"):
            self.base_model = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
            )
        else:
            self.base_model = ChatOllama(
                # model="qwen3:0.6b",
                model="functiongemma:270m",
                temperature=0,
                # seed=10
            )
        self.searcher = EdgeHybridSearch(shard_path)

        self.agent = None

    def get_tools(self):

        @tool
        def list_all_machine(runtime: ExpertToolRuntime) -> list[str]:
            """List names of all machines"""
            return MACHINE_LISTNAME

        @tool
        def search_machine_information(query: str, machine_name: str | None,
                                       runtime: ExpertToolRuntime) -> str:
            """Provide relevant information on machine"""
            if machine_name in ["machine_name", "", None]:
                filter_by = None
            else:
                filter_by = dict(machine_name=machine_name)

            print(f"{query=} {machine_name=}")
            result = self.searcher(query, filter_by=filter_by)
            content = str.join("\n", map(lambda x: x.page_content, result))
            return content

        return [list_all_machine, search_machine_information]

    def init_agent(self, tools: list = [], summarize_history: bool = False):
        if self.agent is None:
            # Agent Creation
            middlewares = []
            if summarize_history:
                middlewares.append(
                    SummarizationMiddleware(
                        model=self.base_model,
                        trigger=dict(fraction=0.7),
                        keep=dict(messages=4)
                    )
                )
            self.agent = create_agent(
                model=self.base_model,
                # system_prompt=SYSTEM_PROMPT,
                state_schema=ExpertState,
                context_schema=ExpertContext,
                response_format=ToolStrategy(ExpertResponse),
                middleware=[
                    define_system_prompt,
                ],
                tools=self.get_tools(),
                checkpointer=self.checkpointer,
                store=self.store
            )

    def __call__(self, user_query: str, thread_id: str | None = None):
        if self.agent is None:
            raise Exception("Agent not intialized")

        thread_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id}}

        print(user_query)

        result = self.agent.invoke(
            dict(
                messages=[
                    HumanMessage(
                        content=user_query
                    )
                ]
            ),
            config=thread_config,
            context=ExpertContext(user_id=thread_id),
        )
        return result
