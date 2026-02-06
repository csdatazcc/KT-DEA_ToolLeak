import numpy as np
from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# --- 初始化加载区 ---
try:
    STOP_WORDS = list(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = list(stopwords.words('english'))

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

print("🔄 正在加载 Embedding 模型 (all-MiniLM-L6-v2)...")
SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
KEYBERT_MODEL = KeyBERT(model=SENTENCE_MODEL)
print("✅ 模型加载完毕！\n")

# 常用颜色 ANSI 转义码
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"  # 重置颜色
# --- --- ---

class ToolSemanticProcessor:
    def __init__(self,
                 tools: List[Dict[str, str]],
                 target_tool_name: str,
                 keybert_top_n: int = 10,
                 keyphrase_ngram_range: Tuple[int, int] = (1, 3),
                 conflict_threshold: float = 0.70): # 阈值：越低越严格，越高越宽松

        if not any(tool['name'] == target_tool_name for tool in tools):
            raise ValueError(f"错误: 目标工具 '{target_tool_name}' 不在工具列表中。")
            
        self.tools_data = {tool['name']: {'description': tool['description']} for tool in tools}
        self.target_tool_name = target_tool_name
        self.non_target_tool_names = [name for name in self.tools_data if name != target_tool_name]

        self.keybert_top_n = keybert_top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.conflict_threshold = conflict_threshold

        self.doc_embeddings = {} 
        for name, data in self.tools_data.items():
            self.doc_embeddings[name] = SENTENCE_MODEL.encode(data['description'], convert_to_tensor=True)

    def _extract_keyphrases(self):
        print("--- 步骤 1: 提取所有工具的关键短语 ---")
        for name, data in self.tools_data.items():
            # KeyBERT 提取
            phrases = KEYBERT_MODEL.extract_keywords(
                data['description'],
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                stop_words=STOP_WORDS,
                use_mmr=True, # Max Marginal Relevance 保证多样性
                diversity=0.3,
                top_n=self.keybert_top_n
            )
            # 过滤掉过短的词 (长度<=2)，保留由 n-gram 产生的有意义短语
            data['phrases'] = [p for p, s in phrases if len(p) > 2]
            
            # 预计算该工具提取出的短语的向量 (加速比对)
            if data['phrases']:
                data['phrase_embeddings'] = SENTENCE_MODEL.encode(data['phrases'], convert_to_tensor=True)
            else:
                data['phrase_embeddings'] = None
            
            # 打印预览
            print(f"  🔹 [{name}] 初步提取 ({len(data['phrases'])}个): {data['phrases'][:3]}...")

    def _find_semantic_conflicts(self):
        print(f"\n--- 步骤 2: 语义冲突扫描 (相似度阈值 > {self.conflict_threshold}) ---")
        
        target_data = self.tools_data[self.target_tool_name]
        if target_data['phrase_embeddings'] is None:
            return []

        conflicts = [] 

        # 扫描所有非目标工具
        for other_name in self.non_target_tool_names:
            other_data = self.tools_data[other_name]
            if other_data['phrase_embeddings'] is None:
                continue

            # ⚡ 矩阵计算: Target短语 x Other短语
            similarity_matrix = util.cos_sim(target_data['phrase_embeddings'], other_data['phrase_embeddings'])

            # 遍历矩阵，找出相似度超标的对子
            for i, target_phrase in enumerate(target_data['phrases']):
                for j, other_phrase in enumerate(other_data['phrases']):
                    score = similarity_matrix[i][j].item()
                    
                    if score > self.conflict_threshold:
                        conflicts.append({
                            "target_phrase": target_phrase,
                            "competitor_phrase": other_phrase,
                            "competitor_tool": other_name,
                            "similarity": score
                        })

        if conflicts:
            print(f"  ⚠️ 发现 {len(conflicts)} 组语义接近的冲突。")
        else:
            print("  ✅ 未发现显著冲突。")
            
        return conflicts

    def _calculate_relevance(self, phrase, tool_name):
        """计算：某个短语 vs 某个工具描述 的语义契合度"""
        phrase_emb = SENTENCE_MODEL.encode(phrase, convert_to_tensor=True)
        doc_emb = self.doc_embeddings[tool_name]
        return util.cos_sim(phrase_emb, doc_emb).item()

    def _resolve_conflicts(self, conflicts):
        print("\n--- 步骤 3: 冲突智能裁决 ---")
        
        # 记录待删除名单 (Tool -> Set of phrases)
        removal_plan = {name: set() for name in self.tools_data}

        for c in conflicts:
            t_phrase = c['target_phrase']
            o_phrase = c['competitor_phrase']
            o_tool = c['competitor_tool']
            
            # 裁判进场：分别计算短语对各自工具的契合度
            score_target = self._calculate_relevance(t_phrase, self.target_tool_name)
            score_competitor = self._calculate_relevance(o_phrase, o_tool)

            print(f"⚔️  冲突: Target['{t_phrase}'] vs {o_tool}['{o_phrase}'] (相似度: {c['similarity']:.2f})")
            
            # 谁的分数低，谁就放弃这个词
            if score_target >= score_competitor:
                print(f"    🏆 目标工具胜出 ({score_target:.3f} vs {score_competitor:.3f})")
                print(f"    🗑️  移除 {o_tool} 的 '{o_phrase}'")
                removal_plan[o_tool].add(o_phrase)
            else:
                print(f"    🛡️ 竞品工具胜出 ({score_competitor:.3f} vs {score_target:.3f})")
                print(f"    🗑️  移除 Target 的 '{t_phrase}'")
                removal_plan[self.target_tool_name].add(t_phrase)

        # 执行删除操作
        for tool_name, phrases_to_remove in removal_plan.items():
            original_list = self.tools_data[tool_name]['phrases']
            self.tools_data[tool_name]['final_phrases'] = [p for p in original_list if p not in phrases_to_remove]

    def process(self):
        self._extract_keyphrases()
        conflicts = self._find_semantic_conflicts()
        
        # 如果有冲突则解决，没冲突则直接复制
        if conflicts:
            self._resolve_conflicts(conflicts)
        else:
            for name, data in self.tools_data.items():
                data['final_phrases'] = data['phrases']

        # 补充逻辑：那些虽然没卷入冲突，但还没生成 final_phrases 字段的工具（比如没有冲突的工具）
        # 需要确保它们也有数据，否则打印会报错
        for name, data in self.tools_data.items():
            if 'final_phrases' not in data:
                 data['final_phrases'] = [p for p in data['phrases'] if p not in self._get_removal_set(name, conflicts)]


        # ✅ 最终：打印所有工具的清单
        print("\n" + "="*40)
        print("🌐 全局最终关键短语清单 (Global Results)")
        print("="*40)
        
        final_results = {}
        for name, data in self.tools_data.items():
            # 排序让输出好看点
            final_list = sorted(data.get('final_phrases', []))
            final_results[name] = final_list

            # 标记是否是目标工具
            if name == self.target_tool_name:
                prefix = f"{GREEN}🎯 TARGET{RESET}"
                name_color = GREEN
            else:
                prefix = f"{BLUE}🔧 TOOL{RESET}"
                name_color = BLUE

            print(f"{prefix}: [{name_color}{name}{RESET}] - 共 {len(final_list)} 个短语")

            if len(final_list) > 0:
                # 每行打印 2 个短语，显得紧凑一点
                for i in range(0, len(final_list), 2):
                    chunk = final_list[i:i+2]
                    print("   • " + "   |   • ".join(chunk))
            else:
                print(f"   {RED}(No Key Phrase!){RESET}")

            print(f"{YELLOW}" + "-" * 20 + f"{RESET}")
            
        return final_results

    def _get_removal_set(self, tool_name, conflicts):
        # 辅助函数：为了处理上面那种没进 resolve 逻辑的边缘情况
        # 在当前逻辑流里其实已经在 resolve 里处理了，这里是为了代码健壮性
        return set()
