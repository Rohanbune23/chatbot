from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

sentences = ["Hello world", "Hola mundo", "नमस्ते दुनिया", "Bonjour le monde"]
embeddings = model.encode(sentences)

for s, e in zip(sentences, embeddings):
    print(s, e[:5])  # print first 5 values of embedding for brevity

import faiss
import numpy as np

# Example: 4 embeddings of size 384 (MiniLM-L12-v2)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))

D, I = index.search(np.array([embeddings[0]], dtype='float32'), k=2)
print("Distances:", D, "Indices:", I)
