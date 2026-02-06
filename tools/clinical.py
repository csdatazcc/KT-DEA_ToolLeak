# tool_clinical_guidelines.py
from tools.base_tools import BaseTool
from typing import Dict, List, Tuple
import difflib

class ClinicalGuidelineTool(BaseTool):
    """
    TOOL 2: Macro-Knowledge (Process Focused)
    Function: Retrieves guidelines using Fuzzy Semantic Matching.
    """

    def __init__(self):
        self._guideline_database = self._load_expanded_guidelines()

    @property
    def name(self) -> str:
        return "ClinicalGuidelineTool"

    @property
    def description(self) -> str:
        return (
            "Useful for providing standard-of-care clinical protocols and guidelines. "
            "Input should be a condition, symptom, or disease name. Returns evidence-based recommendations for clinical decision-making."
        )

    def _calculate_similarity(self, query: str, text: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()

    def run(self, action_input: str) -> str:
        query = action_input.strip()
        print(f"[{self.name}] Searching Clinical Guidelines Repository for: '{query}'")
        
        scored_results: List[Tuple[float, Dict]] = []

        for _, details in self._guideline_database.items():
            s_title = self._calculate_similarity(query, details['title'])
            
            # Weighted Keyword Matching (Simulating Semantic Embeddings)
            keywords_list = details['keywords'].split(',')
            s_keywords = max([self._calculate_similarity(query, k.strip()) for k in keywords_list])
            
            final_score = max(s_title, s_keywords)
            
            if final_score > 0.3:
                scored_results.append((final_score, details))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_results:
            return f"No clinical guidelines found relevant to '{query}'."

        best_score, best_match = scored_results[0]
        
        return (
            f"=== GUIDELINE MATCH (Similarity: {best_score:.2f}) ===\n"
            f"**Title**: {best_match['title']}\n"
            f"**Source**: {best_match['source']}\n"
            f"**Keywords**: {best_match['keywords']}\n"
            f"**Diagnostic Criteria**: {best_match['diagnosis']}\n"
            f"**Treatment Flow**:\n{best_match['treatment_steps']}\n"
            f"**Goals**: {best_match['goals']}"
        )

    def _load_expanded_guidelines(self) -> Dict[str, Dict]:
        """Expanded Database"""
        return {
            # --- Cardiology ---
            "hypertension": {
                "title": "Hypertension Management", "source": "ACC/AHA 2017",
                "keywords": "high blood pressure, htn, elevated bp",
                "diagnosis": "BP >= 130/80 mmHg on >=2 occasions.",
                "treatment_steps": "1. Lifestyle (DASH, Low Na).\n2. First-line: Thiazide, CCB, ACEI/ARB.\n3. Black patients: Thiazide/CCB.\n4. CKD: ACEI/ARB.",
                "goals": "BP < 130/80 mmHg."
            },
            "heart_failure_ref": {
                "title": "Heart Failure (HFrEF) Management", "source": "ACC/AHA/HFSA 2022",
                "keywords": "chf, congestive heart failure, fluid overload, low ejection fraction",
                "diagnosis": "Symptoms (dyspnea, edema) + LVEF <= 40%.",
                "treatment_steps": "1. GDMT 'The Four Pillars':\n   - ARNI (Entresto) or ACEI/ARB.\n   - Beta-blocker (Carvedilol/Metoprolol Succ).\n   - MRA (Spironolactone).\n   - SGLT2 Inhibitor.\n2. Diuretics for volume control.",
                "goals": "Reduce mortality/hospitalization."
            },
            "atrial_fibrillation": {
                "title": "Atrial Fibrillation (AFib)", "source": "ESC 2020",
                "keywords": "afib, arrhythmia, irregular heartbeat",
                "diagnosis": "ECG showing irregular RR intervals, no P waves.",
                "treatment_steps": "1. ABC Pathway:\n   - A: Anticoagulation (CHA2DS2-VASc score).\n   - B: Better symptom control (Rate vs Rhythm control).\n   - C: Comorbidities management.",
                "goals": "Prevent Stroke, Rate < 110 resting."
            },
            "acs_stemi": {
                "title": "Acute STEMI Management", "source": "ESC/AHA",
                "keywords": "heart attack, myocardial infarction, chest pain, st elevation",
                "diagnosis": "ST-segment elevation >1mm in 2 leads + Troponin.",
                "treatment_steps": "1. EMS: Aspirin 325mg, NTG, O2 if sat <90%.\n2. PCI Center: Door-to-Ballon < 90 mins.\n3. Non-PCI Center: Fibrinolytics if transport > 120 mins.\n4. DAPT + Anticoagulant + High-intensity Statin.",
                "goals": "Reperfusion."
            },

            # --- Endocrinology ---
            "diabetes_t2": {
                "title": "Type 2 Diabetes", "source": "ADA 2024",
                "keywords": "t2dm, high sugar, hyperglycemia",
                "diagnosis": "A1c >= 6.5%, FPG >= 126.",
                "treatment_steps": "1. Metformin + Lifestyle.\n2. ASCVD Risk? -> GLP-1 RA or SGLT2i.\n3. CKD/HF? -> SGLT2i.\n4. Need weight loss? -> Tirzepatide/Semaglutide.",
                "goals": "A1c < 7.0%."
            },

            # --- Respiratory ---
            "asthma": {
                "title": "Asthma GINA Guidelines", "source": "GINA 2023",
                "keywords": "wheeze, bronchospasm, reactive airway",
                "diagnosis": "Variable airflow limitation (FEV1 rev >12%).",
                "treatment_steps": "Track 1 (Preferred): ICS-Formoterol as reliever AND maintenance.\nTrack 2: SABA reliever + Daily ICS.",
                "goals": "No exacerbations."
            },
            "copd": {
                "title": "COPD Management (GOLD)", "source": "GOLD 2024",
                "keywords": "emphysema, chronic bronchitis, smoker cough",
                "diagnosis": "Post-bronchodilator FEV1/FVC < 0.70.",
                "treatment_steps": "Group A: Any Bronchodilator.\nGroup B: LABA + LAMA.\nGroup E (Exacerbators): LABA + LAMA (+ ICS if Eosinophils > 300).",
                "goals": "Reduce dyspnea and exacerbations."
            },
            "pneumonia_cap": {
                "title": "Community-Acquired Pneumonia", "source": "ATS/IDSA",
                "keywords": "lung infection, cap, consolidation",
                "diagnosis": "CXR Infiltrate + Fever/Cough.",
                "treatment_steps": "1. Healthy: Amoxicillin or Doxycycline.\n2. Comorbid: Beta-lactam + Macrolide OR Resp FQ (Levofloxacin).",
                "goals": "Clinical stability."
            },

            # --- Neurology ---
            "stroke_ischemic": {
                "title": "Acute Ischemic Stroke", "source": "AHA/ASA",
                "keywords": "cva, stroke, facial droop, slurred speech",
                "diagnosis": "Non-contrast CT Head (rule out bleed).",
                "treatment_steps": "1. tPA (Alteplase/Tenecteplase) if within 4.5 hours and no contraindications.\n2. Mechanical Thrombectomy if Large Vessel Occlusion (up to 24h).\n3. BP control < 185/110 prior to tPA.",
                "goals": "Reperfusion, NIHSS reduction."
            },

            # --- Nephrology ---
            "ckd": {
                "title": "Chronic Kidney Disease (CKD)", "source": "KDIGO 2024",
                "keywords": "renal failure, kidney disease, elevated creatinine",
                "diagnosis": "eGFR < 60 or Albuminuria > 3 months.",
                "treatment_steps": "1. BP Control (ACEI/ARB if albuminuria).\n2. SGLT2 Inhibitors (Dapagliflozin/Empagliflozin).\n3. Statin for lipid control.\n4. Avoid nephrotoxins (NSAIDs).",
                "goals": "Slow progression to ESRD."
            },

            # --- Emergency/Sepsis ---
            "sepsis": {
                "title": "Sepsis & Septic Shock", "source": "SSC 2021",
                "keywords": "infection, hypotension, shock, lactate",
                "diagnosis": "Infection + SOFA >= 2.",
                "treatment_steps": "1. Measure Lactate & Cultures.\n2. Broad spectrum Abx < 1 hour.\n3. 30ml/kg Crystalloid if hypotensive.\n4. Norepinephrine for MAP >= 65.",
                "goals": "MAP > 65 mmHg."
            },
            "anaphylaxis": {
                "title": "Anaphylaxis", "source": "WAO Guidelines",
                "keywords": "allergic reaction, throat closing, hives, shock",
                "diagnosis": "Acute onset skin/mucosa + Resp compromise or Hypotension.",
                "treatment_steps": "1. Epinephrine IM (Mid-thigh) IMMEDIATELY.\n2. Supine position.\n3. IV Fluids.\n4. Adjuncts: Antihistamines, Steroids (Second line).",
                "goals": "Restore airway and perfusion."
            }
        }