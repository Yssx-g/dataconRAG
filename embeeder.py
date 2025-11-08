import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class VectorEmbedder:
    """本地向量化模型调用模块"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为向量"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def batch_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> List[float]:
        """批量计算相似度"""
        similarities = []
        for doc_vec in doc_vecs:
            similarity = self.cosine_similarity(query_vec, doc_vec)
            similarities.append(similarity)
        return similarities
    
    def compute_semantic_diversity(self, text: str, reference_texts: List[str]) -> float:
        """计算文本与参考文本集的语义多样性"""
        if not reference_texts:
            return 1.0
            
        text_vec = self.encode(text)[0]
        ref_vecs = self.encode(reference_texts)
        
        similarities = self.batch_similarity(text_vec, ref_vecs)
        max_similarity = max(similarities) if similarities else 0.0
        
        return 1.0 - max_similarity

test=VectorEmbedder()
texts=["你好啊，我是qwen3","你好啊，我是王世恒"]
print(test.compute_semantic_diversity(texts[0],texts[1]))