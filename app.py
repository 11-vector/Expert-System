import functools as fn
import gradio as gr
from agent import ExpertAgent
from config import VECTOR_DB_DIR

user_id = "worker1"


@fn.lru_cache(maxsize=1)
def load_agent():
    agent = ExpertAgent(shard_path=VECTOR_DB_DIR)
    agent.init_agent()
    return agent


def run_chat(message, history):
    agent = load_agent()
    result = agent(message, user_id, stream_token=True)
    response_msg = ""
    for output in result:
        response_msg += output
        yield response_msg


app = gr.ChatInterface(
    fn=run_chat,
)

agent = load_agent()
agent.searcher('machine')

app.launch(share=True)
