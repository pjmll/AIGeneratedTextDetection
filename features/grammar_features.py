"""
Grammar Feature Extractor for AI Text Detection

"""
import os
import re
import numpy as np
import collections
from typing import List

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ==================== 特征名称 ====================
STAT_FEATURE_NAMES = [
    "grammar_errors",
    "spelling_errors",
    "grammar_rate",
    "spelling_rate",
    "sent_len_var",
    "punct_entropy",
    "word_count"
]

GEC_FEATURE_NAMES = ["rouge2_similarity"]


# ==================== ROUGE-2计算 ====================
def calculate_rouge2_similarity(text1: str, text2: str) -> float:
    """计算两段文本的ROUGE-2 F1分数"""
    def tokenize(text):
        return text.lower().split()
    
    def get_ngrams(tokens, n=2):
        if len(tokens) < n:
            return set()
        return set(zip(*[tokens[i:] for i in range(n)]))
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if len(tokens1) < 2 or len(tokens2) < 2:
        set1, set2 = set(tokens1), set(tokens2)
        if not set1 or not set2:
            return 0.0
        inter = len(set1 & set2)
        p = inter / len(set2)
        r = inter / len(set1)
        return 2 * p * r / (p + r + 1e-10)
    
    ngrams1, ngrams2 = get_ngrams(tokens1), get_ngrams(tokens2)
    if not ngrams1 or not ngrams2:
        return 0.0
    
    inter = len(ngrams1 & ngrams2)
    p = inter / len(ngrams2)
    r = inter / len(ngrams1)
    return 2 * p * r / (p + r + 1e-10)


# ==================== LanguageTool懒加载 ====================
_language_tool_cache = None

def _get_language_tool():
    global _language_tool_cache
    if _language_tool_cache is None:
        try:
            from language_tool_python import LanguageTool
            _language_tool_cache = LanguageTool('en-US')
        except Exception as e:
            print(f"Warning: LanguageTool初始化失败: {e}")
            _language_tool_cache = None
    return _language_tool_cache


# ==================== CoEdit模型接口 ====================
class CoEditModel:
    """CoEdit本地GEC模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import os
            import sentencepiece as spm
            
            abs_path = os.path.abspath(self.model_path)
            print(f"正在加载CoEdit模型: {abs_path}...")
            
            # 1. 验证spiece.model文件是否存在
            sp_model_path = os.path.join(abs_path, "spiece.model")
            if not os.path.exists(sp_model_path):
                print(f"错误: spiece.model不存在于 {abs_path}")
                self.model = None
                return
            
            # 2. 手动加载SentencePiece模型
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model_path)
            print(f"SentencePiece加载成功，词汇表大小: {sp.GetPieceSize()}")
            
            # 3. 加载T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(abs_path)
            
            # 4. 关键步骤：强制替换tokenizer内部的sentencepiece模型
            self.tokenizer.sp_model = sp
            self.tokenizer.sp_model.Load(sp_model_path)
            
            # 5. 加载模型
            self.model = T5ForConditionalGeneration.from_pretrained(abs_path)
            self.model.eval()
            print("CoEdit模型加载成功")
        except Exception as e:
            print(f"CoEdit模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def correct(self, text: str) -> str:
        if self.model is None:
            print("CoEdit模型未加载，返回原文")
            return text
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"CoEdit修正失败: {e}")
            return text


# ==================== 模式1: Statistical ====================
class StatisticalGrammarExtractor:
    """统计模式：使用LanguageTool提取7维特征"""
    
    def __init__(self, use_cache=True):
        self.tool = None
        self.use_cache = use_cache
        self._cache = {} if use_cache else None
    
    def _get_tool(self):
        if self.tool is None:
            self.tool = _get_language_tool()
        return self.tool
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？.!?…]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            words = text.split()
            if len(words) > 20:
                sentences = [' '.join(words[i:i+15]) for i in range(0, len(words), 15)]
        return sentences
    
    def _calculate_punct_entropy(self, text: str) -> float:
        puncs = [c for c in text if c in '.,!?;:\'"()-，。！？；：']
        if not puncs:
            return 0.0
        counts = collections.Counter(puncs)
        probs = np.array(list(counts.values())) / len(puncs)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def extract_features(self, text: str) -> List[float]:
        """提取7维统计特征"""
        if not text or not text.strip():
            return [0.0] * 7
        
        if self.use_cache and text in self._cache:
            return self._cache[text]
        
        tool = self._get_tool()
        grammar_errors, spelling_errors = 0, 0
        
        if tool is not None:
            try:
                for match in tool.check(text):
                    if match.rule_issue_type == 'misspelling':
                        spelling_errors += 1
                    else:
                        grammar_errors += 1
            except Exception:
                pass
        
        word_count = len(text.split())
        grammar_rate = grammar_errors / (word_count + 1)
        spelling_rate = spelling_errors / (word_count + 1)
        
        sentences = self._split_sentences(text)
        sent_len_var = float(np.var([len(s.split()) for s in sentences])) if sentences else 0.0
        punct_entropy = self._calculate_punct_entropy(text)
        
        feats = [
            float(grammar_errors), float(spelling_errors),
            grammar_rate, spelling_rate,
            sent_len_var, punct_entropy,
            float(word_count)
        ]
        
        if self.use_cache:
            self._cache[text] = feats
        return feats
    
    def get_feature_names(self) -> List[str]:
        return STAT_FEATURE_NAMES
    
    def close(self):
        if self.tool is not None:
            try:
                self.tool.close()
            except:
                pass
            self.tool = None


# ==================== 模式2: CoEdit ====================
class CoEditGrammarExtractor:
    """CoEdit模式：使用CoEdit修正+ROUGE-2相似度，提取1维特征"""
    
    def __init__(self, model_path: str, use_cache: bool = True):
        """
        Args:
            model_path: 本地模型文件夹路径，如 "local_models/coedit"
            use_cache: 是否缓存
        """
        self.coedit = CoEditModel(model_path)
        self.use_cache = use_cache
        self._cache = {} if use_cache else None
    
    def extract_features(self, text: str) -> List[float]:
        """提取1维GEC特征（ROUGE-2相似度）"""
        if not text or not text.strip():
            return [0.0]
        
        if self.use_cache and text in self._cache:
            return self._cache[text]
        
        corrected = self.coedit.correct(text)
        similarity = calculate_rouge2_similarity(text, corrected)
        similarity = max(0.0, min(1.0, similarity))
        
        feats = [similarity]
        if self.use_cache:
            self._cache[text] = feats
        return feats
    
    def get_feature_names(self) -> List[str]:
        return GEC_FEATURE_NAMES


# ==================== 工厂函数 ====================
def create_grammar_extractor(mode='statistical', **kwargs):
    """
    创建语法特征提取器
    
    Args:
        mode: 'statistical' 或 'coedit'
        **kwargs: 
            - statistical模式: use_cache=True/False
            - coedit模式: model_path="local_models/coedit", use_cache=True/False
    
    Returns:
        StatisticalGrammarExtractor 或 CoEditGrammarExtractor
    """
    if mode == 'coedit':
        return CoEditGrammarExtractor(**kwargs)
    else:
        return StatisticalGrammarExtractor(**kwargs)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Grammar Feature Extractor - 双模式测试")
    print("=" * 70)
    
    test_texts = [
        "This is a simple test sentense. It hav some error?",
        "The quick brown fox jumps over the lazy dog.",
        "AI text is usually more clean and have less grammar mistakes.",
    ]
    
    # ===== 模式1: Statistical =====
    print("\n[模式1] Statistical (7维)")
    print("-" * 40)
    extractor1 = StatisticalGrammarExtractor()
    for text in test_texts:
        feats = extractor1.extract_features(text)
        print(f"\n{text[:50]}...")
        print(f"  {dict(zip(extractor1.get_feature_names(), feats))}")
    # extractor1.close()
    
    # ===== 模式2: CoEdit =====
    print("\n[模式2] CoEdit (1维 - ROUGE-2相似度)")
    print("-" * 40)
    print("注意: 使用本地模型，请确保已下载到 local_models/coedit 文件夹")
    
    MODEL_PATH = "D:/Project/content_se/AI/project/local_models/coedit"
    
    extractor2 = CoEditGrammarExtractor(model_path=MODEL_PATH)
    for text in test_texts:
        feats = extractor2.extract_features(text)
        print(f"\n原文: {text}")
        corrected = extractor2.coedit.correct(text)
        print(f"修正: {corrected}")
        print(f"  相似度: {feats[0]:.4f}")
    
    print("\n" + "=" * 70)
    print("测试完成")