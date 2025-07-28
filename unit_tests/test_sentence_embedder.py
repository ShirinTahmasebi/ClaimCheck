import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from few_shots_selector.sentence_embedder import MiniLMSentenceEmbedder

sentences = ["This is an example sentence", "Each sentence is converted"]
model = MiniLMSentenceEmbedder()
embeddings = model(sentences)
print(len(embeddings[0])) # What do we expect? 384