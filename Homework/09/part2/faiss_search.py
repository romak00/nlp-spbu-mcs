from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult

class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        Реализовать создание FAISS индекса
        
        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        texts = [f"{doc.title} {doc.text}" for doc in documents]
        embeddings = self.model.encode(texts)
        faiss.normalize_L2(embeddings)
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 1)
        self.index.train(embeddings)
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        """
        Реализовать сохранение индекса
        
        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        with open(path, 'wb') as f:
            ind_ser = faiss.serialize_index(self.index)
            pickle.dump({'documents': self.documents, 'index': ind_ser}, f)

    def load(self, path: str) -> None:
        """
        Реализовать загрузку индекса
        
        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.index = faiss.deserialize_index(data['index'])

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Реализовать поиск
        
        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_embeddings = self.model.encode(query)
        query_embeddings = np.expand_dims(query_embeddings, axis=0)
        faiss.normalize_L2(query_embeddings)
        D, I = self.index.search(query_embeddings, top_k)
        top_docs: List[SearchResult] = []
        for idx, i in enumerate(I.flatten()):
            top_docs.append(SearchResult(
                doc_id=self.documents[i].id,
                score=D.flatten()[idx]/2,
                title=self.documents[i].title,
                text=self.documents[i].text
            ))
        return top_docs

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        Реализовать batch-поиск
        
        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        query_embeddings = self.model.encode(queries)
        faiss.normalize_L2(query_embeddings)
        D, I = self.index.search(query_embeddings, top_k)
        top_docs: List[List[SearchResult]] = []
        top_docs_tmp: List[SearchResult] = []
        for idx, query in enumerate(I):
            for idy, i in enumerate(query):
                top_docs_tmp.append(SearchResult(
                    doc_id=self.documents[i].id,
                    score=D[idx][idy]/2,
                    title=self.documents[i].title,
                    text=self.documents[i].text
                ))
            top_docs.append(top_docs_tmp)
            top_docs_tmp = []
        return top_docs
