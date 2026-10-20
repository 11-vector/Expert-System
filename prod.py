import argparse
from config import VECTOR_DB_DIR
# NOTE: Also vector database used in production actually contains hybrid search
# Hybrid search is actually the combination of dense and sparse search
# using Fusion algotithm such as RRF, Weighed RRF, convex combination, DBSF
# with opptional reranking steps


def index_document():
    from document import load_documents
    import retrevier as rs
    docs = load_documents()
    searcher = rs.EdgeHybridSearch(VECTOR_DB_DIR)
    searcher.index_documents(docs)


def main():
    from agent import ExpertAgent
    user_id = "worker1"
    agent = ExpertAgent(shard_path=VECTOR_DB_DIR)
    agent.init_agent()
    while True:
        query = input("(User)=> ")
        if not query:
            continue
        result = agent(query, user_id)
        if type(result) is str:
            answer = result
        else:
            # answer = result.get('structured_response')
            # if answer is None:
            answer = result.text
        print(f"(Agent)=> {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("expert_cli")
    parser.add_argument(
        "-I", "--index",
        help="Index the document",
        action="store_true"
    )

    args = parser.parse_args()
    if args.index:
        print("Indexing document into VectorDB")
        index_document()
    else:
        main()
