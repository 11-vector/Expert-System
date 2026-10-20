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
    result = agent(message, user_id)
    if result['structured_response']:
        response = result['structured_response']
        answer = f"{response.speech}"\
            f"Current Machine: {response.machine}\n" \
            f"Suggestions: {response.suggestions}\n" \
            f"Confidence: {response.confidence}\n"
    else:
        print(result)
        answer = result['messages'][-1].content

    return answer


app = gr.ChatInterface(
    fn=run_chat,
)

agent = load_agent()
agent.searcher('machine')

app.launch(share=True)
