# agent_framework/tools/__init__.py

from typing import List, Dict, Type
from tools.base_tools import BaseTool
from tools.health_200k import HealthcareRAGTool
from tools.health_200k_dp import HealthcareRAGToolDP
from tools.covid import CovidResearchTool
from tools.clinical import ClinicalGuidelineTool
from tools.drug import DrugReferenceTool
from tools.rag_database import RagDatabase, DPRagDatabase
from tools.rag_system import RAGRetriever, DPRAGRetriever
from tools.corporate import CorporatePolicyTool
from tools.fundamental import FundamentalAccountingTool
from tools.financial import FinancialKnowledgeTool
from tools.law import CivilCodeBM25Tool
from tools.bais import GrepBiasBM25Tool
from tools.criminal import CriminalCodeBM25Tool
from tools.labor import LaborLawBM25Tool
from tools.hate import HateSpeechBM25Tool
from tools.microaggression import MicroaggressionBM25Tool
from tools.pokemon import PokemonDatabaseTool
from tools.email import MarketingEmailTool
from tools.pokemon_move import PokemonMoveTool
from tools.pokemon_item import PokemonItemTool
from tools.phishing import PhishingEmailTool
from tools.HR import HREmailTool
from tools.symptom import SymptomAssessmentBM25Tool
from tools.biomedical import BiomedicalLiteratureBM25Tool
from tools.labresult import LabResultInterpreterBM25Tool


# 可以实例化的工具类列表
# 如果工具需要初始化参数（比如 Wikipedia 的语言），在这里处理
# 例如：wiki_tool_en = WikipediaSearchTool(lang='en')
#      wiki_tool_zh = WikipediaSearchTool(lang='zh')
# 然后在 ALL_TOOLS 中包含需要的实例
_tool_classes: List[Type[BaseTool]] = [
    HealthcareRAGTool,
    CovidResearchTool,
    ClinicalGuidelineTool,
    DrugReferenceTool,
    RagDatabase,
    RAGRetriever,
    CorporatePolicyTool,
    FinancialKnowledgeTool,
    FundamentalAccountingTool,
    CivilCodeBM25Tool,
    GrepBiasBM25Tool,
    CriminalCodeBM25Tool,
    LaborLawBM25Tool,
    HateSpeechBM25Tool,
    MicroaggressionBM25Tool,
    PokemonDatabaseTool,
    MarketingEmailTool,
    PokemonMoveTool,
    PokemonItemTool,
    PhishingEmailTool,
    HREmailTool,
    HealthcareRAGToolDP,
    DPRAGRetriever,
    DPRagDatabase,
    SymptomAssessmentBM25Tool,
    BiomedicalLiteratureBM25Tool,
    LabResultInterpreterBM25Tool
]

# 导出方便外部使用的变量
__all__ = [
   "HealthcareRAGTool",
   "ClinicalGuidelineTool",
   "CovidResearchTool",
   "DrugReferenceTool",
   "RagDatabase",
   "RAGRetriever",
   "CorporatePolicyTool",
   "FinancialKnowledgeTool",
   "FundamentalAccountingTool",
   "CivilCodeBM25Tool",
   "GrepBiasBM25Tool",
   "CriminalCodeBM25Tool",
   "LaborLawBM25Tool",
   "HateSpeechBM25Tool",
   "MicroaggressionBM25Tool",
   "PokemonDatabaseTool",
   "MarketingEmailTool",
   "PokemonMoveTool",
   "PokemonItemTool",
   "PhishingEmailTool",
   "HREmailTool",
   "HealthcareRAGToolDP",
   "DPRAGRetriever",
   "DPRagDatabase",
   "SymptomAssessmentBM25Tool",
   "BiomedicalLiteratureBM25Tool",
   "LabResultInterpreterBM25Tool"
]