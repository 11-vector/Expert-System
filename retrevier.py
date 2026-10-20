# import os
# os.environ['HF_HUB_OFFLINE'] = "1"
import pathlib
import qdrant_edge as q
from langchain_core.documents import Document
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder


class EdgeHybridSearch:

    def __init__(self, shared_dir: str, cache_dir: str = "./cached_model"):
        shared_dir = pathlib.Path(shared_dir)
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        config = q.EdgeConfig(
            vectors=dict(
                dense=q.EdgeVectorParams(
                    size=384,
                    distance=q.Distance.Cosine
                )
            ),
            sparse_vectors=dict(
                sparse=q.EdgeSparseVectorParams(
                    modifier=q.Modifier.Idf)
            ),
        )

        self.sparse_embedding_id = "qdrant/bm25"
        self.dense_embedding_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.reranker_model_id = "cross-encoder/ms-marco-TinyBERT-L2-v2"

        if shared_dir.exists():
            self.shard = q.EdgeShard.load(shared_dir, config)
        else:
            shared_dir.mkdir(parents=True, exist_ok=True)
            self.shard = q.EdgeShard.create(shared_dir, config)

        self.load_models()

    def load_models(self):
        self.sparse_model = q.Bm25(q.Bm25Config(language="english"))
        self.dense_model = TextEmbedding(model_name=self.dense_embedding_id,
                                         cache_dir=self.cache_dir
                                         )
        self.reranker_model = CrossEncoder(
            model_name_or_path=self.reranker_model_id,
            model_kwargs=dict(
                cache_dir=self.cache_dir

            )
        )

    def create_index(self, field: str):
        self.shard.update(
            q.UpdateOperation.create_field_index(
                field,
                q.TextIndexParams(
                    type=q.TextIndexType.TEXT,
                    ascii_folding=True
                )
            )
        )

    def index_documents(self, docs: list[Document], ids: list[int] = None):
        self.shard.update(
            q.UpdateOperation.upsert_points([
                q.Point(
                    id=i,
                    vector=dict(
                        dense=next(self.dense_model.embed(
                            doc.page_content)).tolist(),
                        sparse=self.sparse_model.embed_document(
                            doc.page_content),
                    ),
                    payload=dict(
                        metadata=doc.metadata,
                        page_content=doc.page_content
                    )
                )
                for i, doc in enumerate(docs, start=1)
            ]))

    def __call__(self, query: str, topK: int = 10, gap: int = 5, filter_by: dict[str, str] | None = None):
        topK_with_gap = topK + gap

        filter = q.Filter(
            must=[
                q.FieldCondition(
                    key=key,
                    match=q.MatchText(text=value)
                )
                for (key, value) in filter_by.items()
            ]
        ) if filter_by else None

        prefetch = [
            q.Prefetch(
                query=q.Query.Nearest(
                    query=next(self.dense_model.embed(query)).tolist(),
                    using="dense"
                ),
                filter=filter,
                limit=topK_with_gap
            ),
            q.Prefetch(
                query=q.Query.Nearest(
                    query=self.sparse_model.embed_query(query),
                    using="sparse"
                ),
                filter=filter,
                limit=topK_with_gap
            )
        ]

        result = self.shard.query(
            q.QueryRequest(
                prefetches=prefetch,
                query=q.Fusion.Rrf(k=2, weights=[0.8, 0.2]),
                with_payload=True,
                limit=topK_with_gap
            )
        )

        result = self.rerank(
            query,
            [Document(
                page_content=x.payload.get('page_content'),
                metadata=dict(
                    **x.payload.get('metadata'),
                    score1=x.score
                )
            ) for x in result],
            topK)

        return result

    def rerank(self, query: str, docs: list[Document], topK: int) -> list[Document]:
        result = self.reranker_model.predict(
            [(query, doc.page_content) for doc in docs]
        )
        for i, doc in enumerate(docs):
            doc.metadata['score2'] = result[i]
        sorted_result = sorted(
            docs, key=lambda x: x.metadata['score2'], reverse=True)
        return sorted_result[:topK]
