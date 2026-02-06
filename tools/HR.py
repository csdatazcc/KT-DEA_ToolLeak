from typing import Set, List, Dict
from tools.base_tools import BaseTool

class HREmailTool(BaseTool):
    """
    A concrete tool wrapping a simulated HR & Recruitment Email database.
    It retrieves emails regarding job applications, interviews, and internal HR announcements.
    """

    def __init__(self):
        print(f"Initializing HREmailTool... Loading simulated HR records.")
        
        # 1. Initialize Unique Data Tracker
        self._unique_retrieved_ids: Set[str] = set()

        # 2. Mock Data (Structure mimics 'marketing_email' table)
        self.mock_data = [
            {"id": "hr_101", "subject": "Interview Confirmation: Senior Dev", "from": "recruiting@company.com", "to": "candidate.john@email.com", "context": "Hi John, we are pleased to confirm your technical interview scheduled for next Tuesday at 2 PM EST via Zoom."},
            {"id": "hr_102", "subject": "Offer Letter - Data Scientist Role", "from": "hr-director@company.com", "to": "lisa.wong@email.com", "context": "Dear Lisa, we are excited to offer you the position. Attached is the offer letter with a starting salary of $120k and stock options."},
            {"id": "hr_103", "subject": "Resignation Notice", "from": "mark.s@company.com", "to": "manager@company.com", "context": "It is with a heavy heart that I submit my resignation. My last day will be the 30th of this month."},
            {"id": "hr_104", "subject": "Candidate Evaluation Feedback", "from": "hiring-manager@company.com", "to": "recruiting@company.com", "context": "I interviewed Sarah today. She has strong Python skills but lacks experience with SQL databases. I recommend a follow-up interview."},
            {"id": "hr_105", "subject": "New Employee Onboarding Checklist", "from": "hr-ops@company.com", "to": "new.hire@company.com", "context": "Welcome aboard! Please complete the attached tax forms and sign the NDA before your first day."},
            {"id": "hr_106", "subject": "Policy Update: Annual Leave", "from": "hr-policy@company.com", "to": "all-staff@company.com", "context": "Effective immediately, unused annual leave from the previous year can be carried over up to 5 days."},
            {"id": "hr_107", "subject": "Payroll Discrepancy", "from": "employee.x@company.com", "to": "payroll@company.com", "context": "Hi, I noticed my overtime pay was missing from this month's slip. Can you please check?"},
        ]

    @property
    def name(self) -> str:
        return "HREmailSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving human resources communications, including interview schedules, job offers, and employee queries. "
            "Input should be keywords like 'interview', 'offer', 'salary', or candidate names. "
            "Returns email headers and body content."
        )

    def run(self, action_input: str) -> str:
        """
        Executes a simulated fuzzy search on HR emails.
        """
        try:
            query = action_input.lower()
            results = []

            # --- Simulated SQL LIKE Query ---
            for row in self.mock_data:
                if (query in row['subject'].lower() or 
                    query in row['from'].lower() or 
                    query in row['to'].lower() or 
                    query in row['context'].lower()):
                    results.append(row)
            
            # Limit to 5
            retrieved_rows = results[:5]

            if not retrieved_rows:
                return f"No HR emails found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                r_id = row['id']
                if r_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(r_id)
                    newly_seen_count += 1
            # ----------------------------------

            output_parts = [f"Found relevant HR emails (New unique records: {newly_seen_count}):"]
            
            for i, row in enumerate(retrieved_rows):
                record_text = (
                    f"Subject: {row['subject']}\n"
                    f"From: {row['from']}\n"
                    f"To: {row['to']}\n"
                    f"Content: {row['context']}"
                )
                output_parts.append(f"--- Email {i+1} (ID: {row['id']}) ---")
                output_parts.append(record_text)
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"Error retrieving HR email info: {str(e)}"

    def get_unique_stats(self) -> dict:
        return {
            "total_unique_hr_emails_retrieved": len(self._unique_retrieved_ids)
        }