import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from few_shots_selector.sentence_embedder import SentenceEmbedderWrapper
from helpers.enums import ExampleSelectionPolicy

import faiss

class RagManager:
    def __init__(
        self, 
        sentences_and_labels_list, 
        rag_path, 
        sentence_embedding_model:SentenceEmbedderWrapper
    ):
        self.sentence_embedding_model = sentence_embedding_model
        self.sentences_and_labels_list = sentences_and_labels_list

        if rag_path and os.path.exists(rag_path):
            self.index = faiss.read_index(rag_path)
            print(f"Loaded FAISS index from {rag_path}")
        else:
            print("Building new FAISS index...")
            self.embeddings = self._embed_texts([ex["text"] for ex in sentences_and_labels_list])
            self.index = self._build_faiss_index(self.embeddings)
            if rag_path:
                faiss.write_index(self.index, rag_path)

    def _embed_texts(self, texts):
        return self.sentence_embedding_model(texts)

    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def select(self, input_text, selection_policy: ExampleSelectionPolicy, k=5):
        query_emb = self._embed_texts([input_text])
        examples = []
        labels = []
        scores = []
        if selection_policy == ExampleSelectionPolicy.RANDOM:
            pass
        elif selection_policy == ExampleSelectionPolicy.ONE_PER_CLASS:
            pass
        elif selection_policy == ExampleSelectionPolicy.LEAST_SIMILAR:
            pass
        elif selection_policy == ExampleSelectionPolicy.MOST_SIMILAR:
            s, indices = self.index.search(query_emb, k)
            scores = s[0]
            examples = list(self.sentences_and_labels_list.select(list(indices[0]))["text"])
            labels_code = list(self.sentences_and_labels_list.select(list(indices[0]))["sub_claim_code"])
            labels_text = list(self.sentences_and_labels_list.select(list(indices[0]))["sub_claim"])

        return examples, labels_code, labels_text, scores