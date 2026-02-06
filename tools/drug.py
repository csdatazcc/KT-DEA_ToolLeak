# tool_drug_reference.py
from tools.base_tools import BaseTool
from typing import Dict, List, Tuple
import difflib

class DrugReferenceTool(BaseTool):
    """
    TOOL 1: Micro-Knowledge (Entity Focused)
    Function: Retrieves drug data using Fuzzy Similarity Matching.
    """

    def __init__(self):
        self._drug_database = self._load_expanded_drug_data()

    @property
    def name(self) -> str:
        return "DrugReferenceTool"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving detailed drug information. "
            "Input should be a drug name or drug class. Returns evidence-based drug specifications, interactions, and clinical usage guidance."
        )

    def _calculate_similarity(self, query: str, text: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()

    def run(self, action_input: str) -> str:
        query = action_input.strip()
        print(f"[{self.name}] Searching Pharmaceutical Knowledge Graph for: '{query}'")
        
        scored_results: List[Tuple[float, Dict]] = []

        for _, details in self._drug_database.items():
            # Match against Name, Brand, Class, and Indication (Mechanism keywords)
            score_name = self._calculate_similarity(query, details['name'])
            score_brand = max([self._calculate_similarity(query, b.strip()) for b in details['brand'].split(',')])
            score_class = self._calculate_similarity(query, details['class'])
            
            # Weighted scoring: Name/Brand matches are more important
            final_score = max(score_name, score_brand, score_class * 0.9)
            
            if final_score > 0.35: 
                scored_results.append((final_score, details))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_results:
            return f"No pharmaceutical record found for '{action_input}'."

        # Return Top 1 (Simulate RAG single document retrieval)
        best_score, best_match = scored_results[0]
        
        return (
            f"=== DRUG MONOGRAPH (Similarity: {best_score:.2f}) ===\n"
            f"**Drug**: {best_match['name']} ({best_match['brand']})\n"
            f"**Class**: {best_match['class']}\n"
            f"**Mechanism**: {best_match['mechanism']}\n"
            f"**Dosing**: {best_match['dosing']}\n"
            f"**Side Effects**: {best_match['side_effects']}\n"
            f"**Warnings**: {best_match['warnings']}\n"
            f"**Interactions**: {best_match['interactions']}"
        )

    def _load_expanded_drug_data(self) -> Dict[str, Dict]:
        """Expanded Database"""
        return {
            # --- Endocrine ---
            "metformin": {
                "name": "Metformin", "brand": "Glucophage, Fortamet", "class": "Biguanide",
                "mechanism": "Decreases hepatic glucose production, improves insulin sensitivity.",
                "dosing": "500-2550 mg/day with meals.",
                "side_effects": "GI upset, Lactic Acidosis (rare), B12 deficiency.",
                "warnings": "Renal impairment (eGFR < 30).", "interactions": "Contrast media, Alcohol."
            },
            "levothyroxine": {
                "name": "Levothyroxine", "brand": "Synthroid, Tirosint", "class": "Thyroid Product",
                "mechanism": "Synthetic T4, converts to T3 to regulate metabolism.",
                "dosing": "1.6 mcg/kg/day on empty stomach.",
                "side_effects": "Palpitations, Insomnia, Weight loss.",
                "warnings": "Do not use for weight loss.", "interactions": "Calcium, Iron, PPIs (reduce absorption)."
            },
            
            # --- Cardiovascular ---
            "lisinopril": {
                "name": "Lisinopril", "brand": "Zestril, Prinivil", "class": "ACE Inhibitor",
                "mechanism": "Inhibits ACE, blocks Angiotensin II formation.",
                "dosing": "10-40 mg daily.",
                "side_effects": "Dry cough, Hyperkalemia, Angioedema.",
                "warnings": "Fetal Toxicity (Pregnancy).", "interactions": "NSAIDs, Lithium."
            },
            "atorvastatin": {
                "name": "Atorvastatin", "brand": "Lipitor", "class": "Statin",
                "mechanism": "HMG-CoA Reductase Inhibitor.",
                "dosing": "10-80 mg daily.",
                "side_effects": "Myalgia, Liver enzyme elevation.",
                "warnings": "Rhabdomyolysis.", "interactions": "Grapefruit, Cyclosporine."
            },
            "clopidogrel": {
                "name": "Clopidogrel", "brand": "Plavix", "class": "P2Y12 Inhibitor (Antiplatelet)",
                "mechanism": "Irreversibly blocks P2Y12 component of ADP receptors on platelets.",
                "dosing": "75 mg daily (Loading dose 300-600 mg).",
                "side_effects": "Bleeding, Bruising, TTP.",
                "warnings": "Poor metabolizers (CYP2C19) may have reduced effect.", "interactions": "Omeprazole (reduces effect), NSAIDs."
            },
            "apixaban": {
                "name": "Apixaban", "brand": "Eliquis", "class": "Factor Xa Inhibitor (DOAC)",
                "mechanism": "Selectively inhibits Factor Xa.",
                "dosing": "5 mg BID (AFib); 10 mg BID (DVT tx).",
                "side_effects": "Bleeding, Anemia.",
                "warnings": "Premature discontinuation increases stroke risk.", "interactions": "CYP3A4/P-gp inhibitors."
            },
            "amiodarone": {
                "name": "Amiodarone", "brand": "Pacerone, Cordarone", "class": "Class III Antiarrhythmic",
                "mechanism": "Prolongs action potential duration (K+ channel block) + Na/Ca/Beta blocking effects.",
                "dosing": "Loading 800-1600mg, Maintenance 200-400mg.",
                "side_effects": "Pulmonary fibrosis, Thyroid dysfunction, Blue-gray skin.",
                "warnings": "Lung/Liver toxicity.", "interactions": "Warfarin, Digoxin (increases levels)."
            },

            # --- Respiratory & Allergy ---
            "albuterol": {
                "name": "Albuterol", "brand": "ProAir, Ventolin", "class": "SABA",
                "mechanism": "Beta-2 agonist bronchodilation.",
                "dosing": "2 puffs q4-6h PRN.",
                "side_effects": "Tremor, Tachycardia.",
                "warnings": "Paradoxical bronchospasm.", "interactions": "Beta-blockers."
            },
            "diphenhydramine": {
                "name": "Diphenhydramine", "brand": "Benadryl", "class": "H1 Antagonist (1st Gen)",
                "mechanism": "Competes with histamine for H1-receptor sites.",
                "dosing": "25-50 mg q4-6h.",
                "side_effects": "Sedation, Anticholinergic effects (Dry mouth, Urinary retention).",
                "warnings": "Elderly (Beers Criteria).", "interactions": "Alcohol, CNS depressants."
            },

            # --- Pain & Neurology ---
            "ibuprofen": {
                "name": "Ibuprofen", "brand": "Advil, Motrin", "class": "NSAID",
                "mechanism": "COX-1/COX-2 inhibition.",
                "dosing": "200-800 mg q6-8h.",
                "side_effects": "GI bleed, Kidney injury.",
                "warnings": "CV Thrombotic events.", "interactions": "Anticoagulants, ACEIs."
            },
            "gabapentin": {
                "name": "Gabapentin", "brand": "Neurontin", "class": "Anticonvulsant/Neuropathic Analgesic",
                "mechanism": "Structurally related to GABA; blocks voltage-gated calcium channels.",
                "dosing": "300-600 mg TID.",
                "side_effects": "Dizziness, Somnolence, Peripheral Edema.",
                "warnings": "Respiratory depression with opioids.", "interactions": "Opioids, Antacids."
            },
            "oxycodone": {
                "name": "Oxycodone", "brand": "Roxicodone, OxyContin", "class": "Opioid Agonist",
                "mechanism": "Binds to Mu-opioid receptors in CNS.",
                "dosing": "5-15 mg q4-6h (IR).",
                "side_effects": "Constipation, Respiratory Depression, Sedation, Euphoria.",
                "warnings": "Addiction, Abuse, Misuse (Boxed Warning).", "interactions": "Benzodiazepines, Alcohol."
            },

            # --- Emergency / ACLS ---
            "epinephrine": {
                "name": "Epinephrine", "brand": "Adrenalin, EpiPen", "class": "Sympathomimetic",
                "mechanism": "Alpha/Beta agonist (Vasoconstriction + Bronchodilation + Inotropy).",
                "dosing": "Cardiac Arrest: 1 mg IV q3-5min. Anaphylaxis: 0.3-0.5 mg IM.",
                "side_effects": "Palpitations, Anxiety, HTN crisis.",
                "warnings": "Extravasation necrosis.", "interactions": "Beta-blockers."
            },
            
            # --- Anti-Infective ---
            "amoxicillin": {
                "name": "Amoxicillin", "brand": "Amoxil", "class": "Penicillin",
                "mechanism": "Cell wall synthesis inhibitor.",
                "dosing": "500-875 mg q12h.",
                "side_effects": "Rash, Diarrhea.",
                "warnings": "Anaphylaxis.", "interactions": "Warfarin."
            },
            "ciprofloxacin": {
                "name": "Ciprofloxacin", "brand": "Cipro", "class": "Fluoroquinolone",
                "mechanism": "Inhibits DNA Gyrase/Topoisomerase IV.",
                "dosing": "250-750 mg q12h.",
                "side_effects": "Tendonitis, QTc prolongation.",
                "warnings": "Tendon rupture, Aortic dissection.", "interactions": "Tizanidine, Multivitamins (absorption)."
            },

            # --- Gastrointestinal ---
            "ondansetron": {
                "name": "Ondansetron", "brand": "Zofran", "class": "5-HT3 Antagonist",
                "mechanism": "Blocks serotonin receptors in CTZ and vagal nerve terminals.",
                "dosing": "4-8 mg q8h PRN.",
                "side_effects": "Headache, Constipation, QTc prolongation.",
                "warnings": "Serotonin Syndrome.", "interactions": "Apomorphine, SSRIs."
            }
        }