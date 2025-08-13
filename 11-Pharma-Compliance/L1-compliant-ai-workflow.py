# --- 
jupyter:
  jupytext:
    text_representation:
      extension: .py
      format_name: light
      format_version: '1.5'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
# ---

# # Lab: Build a Compliant AI Workflow for QC Reports
# 
# **Goal:** In this lab, you will build a simplified but compliant workflow for analyzing a lab report using an AI model. The focus is not on the AI's accuracy, but on the **governance and traceability** required in a regulated GxP environment. 
# 
# **Key Concepts:**
# - **Audit Trails:** Understanding the "who, what, when, and why" of every action.
# - **Versioning:** Treating prompts and models as controlled documents.
# - **Human-in-the-Loop (HITL):** Ensuring a qualified person makes the final decision. 
# - **Electronic Signatures:** Securely associating a person with an action. 
# 
# ---
# 
# ## 1. Setup
# 
# We'll need pandas to interact with our CSV-based audit log.

# + 
import pandas as pd
from datetime import datetime
import hashlib
import getpass  # To simulate getting the current user

# Define the path to our audit log
AUDIT_LOG_PATH = "data/audit_log.csv"
# - 

# ## 2. The Scenario
# 
# Imagine you are a QC analyst. A new lab report has come in for Batch `AP-2024-03-25-B004`. Your task is to use an AI assistant to analyze this report for potential issues and then formally approve or reject the AI's analysis. 
# 
# Every significant action you take must be logged in the `audit_log.csv` file, creating an unbreakable chain of evidence that would satisfy an auditor. 
# 
# ### Our "Database": The Audit Log
# Let's create a helper function to write events to our CSV file. In a real system, this would be a validated database that complies with 21 CFR Part 11.

# + 
def log_event(event_type: str, user: str, prompt_version: str = "N/A", batch_id: str = "N/A", details: str = "", signature: str = "N/A"):
    """
    Appends a new event to the audit log.
    """
    try:
        # Create a new record
        new_log_entry = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user": user,
            "prompt_version": prompt_version,
            "batch_id": batch_id,
            "details": details,
            "signature": signature
        }])
        
        # Append to the CSV file
        new_log_entry.to_csv(AUDIT_LOG_PATH, mode='a', header=False, index=False)
        print(f"Event logged: {event_type}")
    except Exception as e:
        print(f"Failed to log event: {e}")

# Let's simulate a user login event
current_user = getpass.getuser()
log_event("USER_LOGIN", user=current_user, details="User logged into the system.")
# - 

# ## 3. Versioning the Prompt
# 
# In a compliant system, you can't just use any prompt. The prompt itself must be version-controlled. A change to the prompt could change the AI's output, so auditors need to know *exactly* which prompt was used. 
# 
# We will simulate this by creating a "Prompt Library." In a real system, this would be a validated document management system or a Git repository.

# + 
# Our "Prompt Library"
PROMPT_LIBRARY = {
    "v1.0.0": "Analyze the following lab report and summarize any potential deviations or out-of-spec results.",
    "v1.1.0": "Analyze the following lab report. Summarize any potential deviations or out-of-spec results. Also, highlight any results that are within 5% of their specification limit."
}

# We will use the latest, approved version for our analysis.
ACTIVE_PROMPT_VERSION = "v1.1.0"
ACTIVE_PROMPT = PROMPT_LIBRARY[ACTIVE_PROMPT_VERSION]

print(f"Using Prompt Version: {ACTIVE_PROMPT_VERSION}")
print(f"Prompt Text: '{ACTIVE_PROMPT}'")
# - 

# ## 4. Simulating the AI Analysis
# 
# Now, we'll run our "AI analysis." For this lab, we will **simulate** the AI's output to focus on the workflow. In a real application, you would call the LLM here, similar to the previous lab. 
# 
# The key is that we **log the interaction**. We record who ran the analysis, on which batch, and with which prompt version.

# + 
# Load the lab report we want to analyze
try:
    with open("../10-Pharma-QC/data/lab_report_04.txt", "r") as f:
        lab_report_text = f.read()
    BATCH_ID = "AP-2024-03-25-B004"
    print(f"Loaded report for batch: {BATCH_ID}")
except FileNotFoundError:
    print("Error: Make sure the lab report from the previous lab exists at '../10-Pharma-QC/data/lab_report_04.txt'")
    lab_report_text = ""

# --- SIMULATED AI OUTPUT ---
# In a real lab, you would call your LLM here with the ACTIVE_PROMPT and lab_report_text.
ai_analysis_result = {
    "summary": "The batch meets all specifications.",
    "details": "However, the Dissolution Test result (81%) is within 5% of its lower specification limit (80%). The Salicylic Acid impurity (0.09%) is also close to its limit (<= 0.1%).",
    "recommendation": "FLAG FOR HUMAN REVIEW"
}
# --- END SIMULATION ---

# Log the AI analysis event
log_event(
    event_type="AI_ANALYSIS_RUN",
    user=current_user,
    prompt_version=ACTIVE_PROMPT_VERSION,
    batch_id=BATCH_ID,
    details=f"AI recommendation: {ai_analysis_result['recommendation']}. Summary: {ai_analysis_result['summary']}"
)

print("\nAI Analysis Complete. Output:")
print(ai_analysis_result)
# - 

# ## 5. Human-in-the-Loop: Review and Signature
# 
# The AI has made a recommendation: "FLAG FOR HUMAN REVIEW." It has not made the final decision. Now, a human analyst must review the AI's findings and make the final call. This is the **Human-in-the-Loop (HITL)** step. 
# 
# To make a final decision, the analyst must provide their "electronic signature." In a real 21 CFR Part 11 system, this involves re-entering your password. We will simulate this by creating a hash of the user's password combined with the data being signed.

# + 
def create_signature(user: str, password_input: str, data_to_sign: str) -> str:
    """
    Creates a simulated electronic signature.
    In a real system, you would use a secure, validated method.
    """
    # Combine user, password, and data to create a unique signature hash
    signature_string = f"{user}:{password_input}:{data_to_sign}"
    return hashlib.sha256(signature_string.encode()).hexdigest()

# The analyst reviews the AI's output and makes a decision.
human_decision = "I agree with the AI. The results are close to the limits. I will flag this for further investigation."
final_classification = "FLAGGED"

# The analyst must "sign" their decision.
# getpass.getpass() securely prompts for a password without showing it on the screen.
password = getpass.getpass(prompt=f"User '{current_user}', please enter your password to sign: ")

# We sign the combination of the batch ID and the final classification.
data_being_signed = f"{BATCH_ID}:{final_classification}"
user_signature = create_signature(current_user, password, data_being_signed)

print(f"\nGenerated Signature: {user_signature}")

# Log the final human decision with the signature
log_event(
    event_type="HUMAN_APPROVAL",
    user=current_user,
    batch_id=BATCH_ID,
    details=f"Final Classification: {final_classification}. Justification: {human_decision}",
    signature=user_signature
)
# - 

# ## 6. Reviewing the Audit Trail
# 
# The process is complete. Let's now act as an auditor and review the `audit_log.csv` file. 
# 
# A complete audit trail should tell a clear, chronological story of everything that happened.

# + 
try:
    audit_df = pd.read_csv(AUDIT_LOG_PATH)
    print("--- Audit Log Review ---")
    # Display the full log for review
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    print(audit_df)
except FileNotFoundError:
    print("Audit log not found.")
except Exception as e:
    print(f"Could not read audit log: {e}")
# - 

# ### Exercise: Interpreting the Log
# 
# **Your Task:** Look at the audit log above and answer the following questions, as if you were an auditor. 
# 
# 1.  Who performed the final review of batch `AP-2024-03-25-B004`?
# 2.  What version of the analysis prompt was used?
# 3.  What was the final, human-approved classification for the batch?
# 4.  Is the entire process from login to final signature captured?
# 
# This exercise demonstrates how a well-maintained audit trail provides the traceability required for regulatory compliance.
# 
# ---
# 
# ## Conclusion
# 
# In this lab, you did not build a complex AI, but you performed an even more critical task: you built a **compliant process** around a simple AI. 
# 
# You have learned how to:
# 1.  Create and maintain a detailed audit log for every system event.
# 2.  Incorporate prompt versioning into your workflow.
# 3.  Log the specific details of an AI analysis for traceability.
# 4.  Implement a Human-in-the-Loop (HITL) decision step.
# 5.  Generate and record a simulated electronic signature to ensure accountability.
# 
# These governance principles are essential for deploying any AI system in a regulated pharmaceutical environment.
