import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class LegalRetriever:
    def __init__(self, json_file_path="korean_optimized_legal.json"):
        self.data = self.load_data(json_file_path)
        if self.data:
            vector_dim = len(self.data[0]["vector"])
            model_map = {384: "sentence-transformers/all-MiniLM-L6-v2", 768: "jhgan/ko-sroberta-multitask"}
            self.model = SentenceTransformer(model_map.get(vector_dim, "jhgan/ko-sroberta-multitask"))
    
    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def search(self, query, top_k=3):
        if not self.data or not query:
            return []
        
        query_vector = self.model.encode([query], convert_to_tensor=False)[0]
        results = []
        
        for item in self.data:
            stored_vector = np.array(item["vector"])
            similarity = cosine_similarity([query_vector], [stored_vector])[0][0]
            
            results.append({
                "text": item["text"],
                "similarity": round(similarity, 4),
                "metadata": item.get("metadata", {})
            })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
    
    def run(self):
        while True:
            query = input("\n질문: ").strip()
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                break
            
            results = self.search(query)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. ({result['similarity']})")
                
                text = result["text"]
                if len(text) > 300:
                    last_period = text.rfind('.', 0, 300)
                    if last_period > 200:
                        text = text[:last_period + 1]
                    else:
                        text = text[:300]
                    text += "..."
                
                print(text)
                
                meta = result["metadata"]
                if meta.get("조_번호"):
                    print(f"제{meta['조_번호']}조")

if __name__ == "__main__":
    retriever = LegalRetriever()
    retriever.run()