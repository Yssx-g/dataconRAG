# qwen_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import re

class QwenModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype='auto',
            device_map="auto"
        )
        
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7,system_set: str="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> str:
        """生成响应"""
        messages = [
            {"role": "system", "content": system_set },
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def batch_generate(self, prompts: List[str], max_length: int = 512) -> List[str]:
        """批量生成响应"""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_length)
            responses.append(response)
        return responses

class AnchorExtractor:
    """智能锚点提取器 - 使用Qwen模型"""
    
    def __init__(self, qwen_model: QwenModel):
        self.model = qwen_model
        
    def extract_anchors_intelligent(self, text: str, max_anchors: int = 5) -> List[str]:
        """智能提取锚点"""
        prompt = f"""
请从以下文本中最多提取{max_anchors}个重要的关键词或关键短语作为锚点。这些锚点应该能够代表文本的核心内容，并且适合用于生成相关的查询问题。

文本内容：
{text}

请直接返回锚点列表，每个锚点用换行符分隔，不要添加任何解释。
"""
        

        response = self.model.generate_response(prompt, max_length=200, temperature=0.3)
        anchors = self._parse_anchor_response(response)
        return anchors[:max_anchors]

    
    def _parse_anchor_response(self, response: str) -> List[str]:
        """解析模型返回的锚点"""
        anchors = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 移除编号和特殊字符
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            if line and len(line) > 1 and len(line) < 50:  # 合理的锚点长度
                anchors.append(line)
        
        return anchors

# test=AnchorExtractor(local_model)
# print(test.extract_anchors_intelligent(texts,100))

class SentenceExpander:
    """智能句子扩展器 - 使用Qwen模型"""

    def __init__(self, qwen_model: QwenModel,overlap_ratio):
        self.model = qwen_model
        self.overlap_ratio = overlap_ratio #重叠率

    def generate_diversified_queries(self, chunk: str) -> Dict[str, List[str]]:
        """
        基于chunk生成多样化的查询
        返回包含3个前向查询、3个后向查询和2个重叠块查询的字典
        """
        queries = {
            "forward": [],
            "backward": [],
            "overlap": []
        }
        # 生成前向查询
        queries["forward"] = self._generate_ward_queries(chunk, 3,'forward')       
        # 生成后向查询
        queries["backward"] = self._generate_ward_queries(chunk, 3,'backward')     
        # 生成重叠块查询
        queries["overlap"] = self._generate_overlap_queries(chunk, 2)    
        return queries
    
    def _generate_ward_queries(self, chunk: str, num_queries: int,mode: str) -> List[str]:
        """生成查询"""
        pro=[]
        pro.append( f"""
基于以下文本内容，生成{num_queries}个不同的前向查询问题。这些问题应该：
1. 询问文本后续可能的内容或发展
2. 与文本内容高度相关且自然
3. 适合用于信息检索

文本内容：
{chunk}

请直接返回{num_queries}个查询问题，每个问题用换行符分隔，不要添加任何解释或编号。
""")
        pro.append(f"""
基于以下文本内容，生成{num_queries}个不同的后向查询问题。这些问题应该：
1. 询问文本之前可能的内容或背景
2. 与文本内容高度相关且自然
3. 适合用于信息检索

文本内容：
{chunk}

请直接返回{num_queries}个查询问题，每个问题用换行符分隔，不要添加任何解释或编号。
""")
        if mode=='forward':
            prompt=pro[0]
            response = self.model.generate_response(prompt, max_length=200, temperature=0.7)
            queries = self._parse_query_response(response)
            return queries[:num_queries]
        elif mode=='backward':
            prompt=pro[1]   # wrong
            response = self.model.generate_response(prompt, max_length=200, temperature=0.7)
            queries = self._parse_query_response(response)
            return queries[:num_queries]
        else:
            print("Error paramater in _generate_ward_queries!")
            exit()

    def _generate_overlap_queries(self, chunk: str, num_queries: int) -> List[str]:
        """生成重叠块查询"""
        # 计算重叠部分
        chunk_length = len(chunk)
        overlap_length = int(chunk_length * self.overlap_ratio)
        
        # 获取开头和结尾的重叠部分
        start_overlap = chunk[:overlap_length] if chunk_length > overlap_length else chunk
        end_overlap = chunk[-overlap_length:] if chunk_length > overlap_length else chunk
        
        queries = []
        
        # 为开头重叠部分生成查询
        if start_overlap:
            prompt = f"""
基于以下文本片段，生成{num_queries//2}个查询问题。这些查询应该：
1. 基于文本片段的开头部分
2. 能够帮助检索到与这个开头相关的其他内容
3. 自然且适合用于信息检索

文本片段（开头）：
{start_overlap}

请直接返回{num_queries//2}个查询问题，每个问题用换行符分隔。
""" 
            response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
            start_queries = self._parse_query_response(response)
            queries.extend(start_queries[:num_queries//2])
        
        # 为结尾重叠部分生成查询
        if end_overlap and len(queries) < num_queries:
            prompt = f"""
基于以下文本片段，生成{num_queries - len(queries)}个查询问题。这些查询应该：
1. 基于文本片段的结尾部分
2. 能够帮助检索到与这个结尾相关的其他内容
3. 自然且适合用于信息检索

文本片段（结尾）：
{end_overlap}

请直接返回{num_queries - len(queries)}个查询问题，每个问题用换行符分隔。
"""
            response = self.model.generate_response(prompt, max_length=150, temperature=0.6)
            end_queries = self._parse_query_response(response)
            queries.extend(end_queries[:num_queries - len(queries)])
        return queries[:num_queries]

    def _parse_query_response(self, response: str) -> List[str]:
        """解析模型返回的查询"""
        queries = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # 移除编号和特殊字符
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'^["\']|["\']$', '', line)  # 移除引号
            
            if line and len(line) > 8 and len(line) < 150:  # 合理的查询长度
                queries.append(line)
        
        return queries

if __name__ == "__main__":
    texts = "老屋的门轴在风里生锈了，我推门时听见它发出悠长的叹息。檐角垂下的蛛网在夕阳里泛着金边，像母亲织了一半的毛衣。院中的槐树还在，只是枝干裂开的缝隙里，爬满了时间的青苔。父亲蹲在门槛上抽旱烟，烟锅里火星明灭，仿佛在数着归家的路。\
石板路被岁月磨得发亮，踩上去像踏着旧日的琴键。母亲从厨房探出头，围裙上的油渍依旧晕染着当年的形状。她捧着搪瓷碗的手在颤抖，碗底沉着的鸡蛋黄，是三十年前我离家时她塞进我行囊的形状。巷口的蝉鸣突然静了，只剩下竹床在院中吱呀作响，惊起一地碎金。"
    local_model=QwenModel()
    test2=SentenceExpander(local_model,0.2)
    res= test2.generate_diversified_queries(texts)
    [print(item) for item in res['forward']]
    [print(item) for item in res['backward']]
    [print(item) for item in res['overlap']]
