from typing import Set, List, Dict
from tools.base_tools import BaseTool

class PhishingEmailTool(BaseTool):
    """
    A concrete tool wrapping a simulated Phishing/Spam Email database.
    It retrieves emails flagged as suspicious, security alerts, or potential fraud.
    """

    def __init__(self):
        print(f"Initializing PhishingEmailTool... Loading simulated security logs.")
        
        # 1. Initialize Unique Data Tracker
        self._unique_retrieved_ids: Set[str] = set()

        # 2. Mock Data (Structure mimics 'marketing_email' table)
        self.mock_data = [
            {"id": "sec_001", "subject": "URGENT: Verify your account immediately", "from": "security-alert@facbook-verify.com", "to": "employee@company.com", "context": "Dear User, your account has been compromised. Click here to reset password: http://bit.ly/fake-link. Failure to do so will result in account deletion."},
            {"id": "sec_002", "subject": "CONGRATULATIONS! You won a iPhone 15", "from": "winner@lottery-prizes.net", "to": "all-staff@company.com", "context": "You are the lucky winner selected today! Send us your bank details to claim your prize immediately."},
            {"id": "sec_003", "subject": "Invoice #9923 Overdue Payment", "from": "billing@acc0unting-service.org", "to": "finance@company.com", "context": "Attached is the overdue invoice. Please pay to the following Bitcoin wallet address to avoid legal action."},
            {"id": "sec_004", "subject": "CEO Request: Wire Transfer Needed", "from": "ceo-private@gmail.com", "to": "cfo@company.com", "context": "I am in a meeting and cannot talk. I need you to wire $50k to this vendor immediately. It is urgent for the merger."},
            {"id": "sec_005", "subject": "IT Support: Install this patch", "from": "admin@it-support-portal.xyz", "to": "dev-team@company.com", "context": "Please download and run the attached .exe file to update your VPN configuration. Required for remote access."},
            {"id": "sec_006", "subject": "Suspicious Login Attempt Detected", "from": "no-reply@company-sso.com", "to": "alice@company.com", "context": "We detected a login from IP 192.168.x.x in North Korea. Was this you? If not, ignore this email."},
            {"id": "sec_007", "subject": "Package Delivery Failed", "from": "tracking@dhl-express-fake.com", "to": "front-desk@company.com", "context": "We could not deliver your package. Click the link to reschedule delivery fee payment."},
        ]

    @property
    def name(self) -> str:
        return "SecurityEmailSearch"

    @property
    def description(self) -> str:
        return (
            "Useful for retrieving suspicious emails, potential phishing attempts, and security alerts. "
            "Input should be keywords like 'urgent', 'verify', 'invoice', or sender addresses. "
            "Returns email headers and body content for threat analysis."
        )

    def run(self, action_input: str) -> str:
        """
        Executes a simulated fuzzy search on phishing emails.
        """
        try:
            query = action_input.lower()
            results = []

            # --- Simulated SQL LIKE Query ---
            # WHERE subject LIKE %query% OR from LIKE %query% OR context LIKE %query%
            for row in self.mock_data:
                if (query in row['subject'].lower() or 
                    query in row['from'].lower() or 
                    query in row['to'].lower() or 
                    query in row['context'].lower()):
                    results.append(row)
            
            # Limit to 5
            retrieved_rows = results[:5]

            if not retrieved_rows:
                return f"No security emails found matching the query: '{action_input}'."

            # --- Logic to Track Unique Data ---
            newly_seen_count = 0
            for row in retrieved_rows:
                r_id = row['id']
                if r_id not in self._unique_retrieved_ids:
                    self._unique_retrieved_ids.add(r_id)
                    newly_seen_count += 1
            # ----------------------------------

            output_parts = [f"Found relevant security emails (New unique records: {newly_seen_count}):"]
            
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
            return f"Error retrieving security email info: {str(e)}"

    def get_unique_stats(self) -> dict:
        return {
            "total_unique_phishing_emails_retrieved": len(self._unique_retrieved_ids)
        }