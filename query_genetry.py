# cache_manager.py
import numpy as np
from typing import List, Dict, Any, Tuple
from embeeder import VectorEmbedder
from qwen_model import AnchorExtractor, SentenceExpander
from rag_client import RAGAttackClient

from chat import askanythingLLM

class CacheManager:
    """缓存管理器 - 管理短期和长期缓存"""

    def __init__(self, 
                 qwen_model: Any,
                 similarity_threshold: float = 0.8,
                 overlap_ratio: float = 0.2):
        """
        初始化缓存管理器
        
        Args:
            qwen_model: Qwen模型实例
            similarity_threshold: 相似度阈值，高于此值则舍去
            overlap_ratio: 重叠率
        """
        # 初始化向量化模型
        self.embedder = VectorEmbedder()

        # 初始化锚点提取器和句子扩展器
        self.anchor_extractor = AnchorExtractor(qwen_model)
        self.sentence_expander = SentenceExpander(qwen_model, overlap_ratio)

        # 初始化RAG客户端
        # self.rag_client = RAGAttackClient()
        self.rag_client = askanythingLLM()

        # 缓存设置
        self.similarity_threshold = similarity_threshold

        # 初始化缓存
        self.short_term_anchor_cache = []  # 短期锚点缓存 [(text, vector), ...]
        self.long_term_anchor_cache = []   # 长期锚点缓存 [(text, vector), ...]

        self.short_term_sentence_cache = []  # 短期语句缓存 [(text, vector), ...]
        self.long_term_sentence_cache = []   # 长期语句缓存 [(text, vector), ...]

        # 存储RAG查询结果
        self.rag_results = []

    def add_to_anchor_cache(self, text: str) -> bool:
        """
        将锚点文本添加到缓存中
        
        Returns:
            bool: 是否成功添加（相似度检查通过）
        """

        # 向量化文本
        vector = self.embedder.encode(text)[0]

        # 检查与长期缓存的相似度
        if self.long_term_anchor_cache:
            max_similarity = self._get_max_similarity(vector, self.long_term_anchor_cache)
            if max_similarity > self.similarity_threshold:
                return False  # 相似度过高，舍去

        # 添加到短期缓存和长期缓存
        self.short_term_anchor_cache.append((text, vector))
        self.long_term_anchor_cache.append((text, vector))

        return True

    def add_to_sentence_cache(self, text: str) -> bool:
        """
        将语句文本添加到缓存中
        
        Returns:
            bool: 是否成功添加（相似度检查通过）
        """
        # 向量化文本
        vector = self.embedder.encode(text)[0]

        # 检查与长期缓存的相似度
        if self.long_term_sentence_cache:
            max_similarity = self._get_max_similarity(vector, self.long_term_sentence_cache)
            if max_similarity > self.similarity_threshold:
                return False  # 相似度过高，舍去

        # 添加到短期缓存和长期缓存
        self.short_term_sentence_cache.append((text, vector))
        self.long_term_sentence_cache.append((text, vector))

        return True

    def _get_max_similarity(self, query_vector: np.ndarray, cache: List[Tuple[str, np.ndarray]]) -> float:
        """计算查询向量与缓存中所有向量的最大相似度"""
        max_similarity = 0.0
        for _, cached_vector in cache:
            similarity = self.embedder.cosine_similarity(query_vector, cached_vector)
            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def generate_and_cache_anchors(self, text: str, max_anchors: int = 5) -> List[str]:
        """生成锚点并添加到缓存"""
        anchors = self.anchor_extractor.extract_anchors_intelligent(text, max_anchors)

        added_anchors = []
        for anchor in anchors:
            if self.add_to_anchor_cache(anchor):
                added_anchors.append(anchor)

        return added_anchors

    def generate_and_cache_queries(self, chunk: str) -> Dict[str, List[str]]:
        """生成多样化查询并添加到缓存"""
        queries_dict = self.sentence_expander.generate_diversified_queries(chunk)

        # 合并所有查询
        all_queries = []
        for query_type, queries in queries_dict.items():
            all_queries.extend(queries)

        # 添加到缓存
        added_queries = []
        for query in all_queries:
            if self.add_to_sentence_cache(query):
                added_queries.append(query)

        return queries_dict

    def query_short_term_cache(self) -> List[Dict[str, Any]]:
        """查询短期缓存中的所有内容"""
        results = []

        # 查询短期锚点缓存
        for text, _ in self.short_term_anchor_cache:
            rag_result = self.rag_client.query(text)
            results.append({
                'type': 'anchor',
                'query': text,
                'response': rag_result.get('output', ''),
                'full_result': rag_result
            })
            self.rag_results.append(results[-1])

        # 查询短期语句缓存
        for text, _ in self.short_term_sentence_cache:
            rag_result = self.rag_client.query(text)
            results.append({
                'type': 'sentence',
                'query': text,
                'response': rag_result.get('output', ''),
                'full_result': rag_result
            })
            self.rag_results.append(results[-1])

        # 清空短期缓存（可选，根据需求决定）
        # self.clear_short_term_cache()

        return results

    def clear_short_term_cache(self):
        """清空短期缓存"""
        self.short_term_anchor_cache.clear()
        self.short_term_sentence_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'short_term_anchors': len(self.short_term_anchor_cache),
            'long_term_anchors': len(self.long_term_anchor_cache),
            'short_term_sentences': len(self.short_term_sentence_cache),
            'long_term_sentences': len(self.long_term_sentence_cache),
            'rag_results': len(self.rag_results)
        }

    def process_text_comprehensive(self, text: str, ) -> Dict[str, Any]:
        """
        综合处理文本：生成锚点、查询，并执行RAG查询
        
        Returns:
            包含所有结果的字典
        """
        # 生成并缓存锚点
        anchors = self.generate_and_cache_anchors(text)
        print("anchors:",anchors)
        # 生成并缓存查询
        queries = self.generate_and_cache_queries(text)
        print("queries",queries)
        # 执行RAG查询
        # rag_results = self.query_short_term_cache()

        return {
            'anchors_generated': anchors,
            'queries_generated': queries,
            # 'rag_results': rag_results,
            'cache_stats': self.get_cache_stats()
        }


# 使用示例
if __name__ == "__main__":
    from qwen_model import QwenModel
    
    # 初始化Qwen模型
    qwen_model = QwenModel()
    
    # 初始化缓存管理器
    cache_manager = CacheManager(qwen_model, similarity_threshold=0.75)
    
    # 测试文本
    test_text = """
    老屋的门轴在风里生锈了，我推门时听见它发出悠长的叹息。檐角垂下的蛛网在夕阳里泛着金边，像母亲织了一半的毛衣。
    院中的槐树还在，只是枝干裂开的缝隙里，爬满了时间的青苔。父亲蹲在门槛上抽旱烟，烟锅里火星明灭，仿佛在数着归家的路。
    """
    
    # 综合处理文本
    results = cache_manager.process_text_comprehensive(test_text)
    
    # print("=== 生成的锚点 ===")
    # for anchor in results['anchors_generated']:
    #     print(f"- {anchor}")
    
    # print("\n=== 生成的查询 ===")
    # for query_type, queries in results['queries_generated'].items():
    #     print(f"{query_type}:")
    #     for query in queries:
    #         print(f"  - {query}")
    
    # print("\n=== RAG查询结果 ===")
    # for i, result in enumerate(results['rag_results']):
    #     print(f"{i+1}. [{result['type']}] {result['query']}")
    #     print(f"   响应: {result['response'][:100]}...")
    
    # print(f"\n=== 缓存统计 ===")
    # stats = results['cache_stats']
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
