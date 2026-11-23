"""
Augmentation Techniques
=======================
Defines the five augmentation technique prompts used with Gemini API
and also provides demo (fallback) variations for each technique.
"""

from config.settings import AUGMENTATION_TECHNIQUES

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Prompt Templates
# Each template must include {clause} and {category} placeholders.
# ─────────────────────────────────────────────────────────────────────────────

TECHNIQUE_PROMPTS = {
    "Legal Paraphrasing": """
You are an expert legal drafting assistant.

Your task is to paraphrase the following contract clause using different legal expressions.

STRICT RULES:
- Preserve ALL legal obligations, rights, amounts, dates, parties, conditions, and exceptions EXACTLY.
- Do NOT invent new legal terms or add new contractual obligations.
- Maintain the original clause category: {category}.
- The rewritten clause must be clearly a paraphrase — not a copy.
- Output ONLY the rewritten clause text. No explanations, no labels, no preamble.

Original Clause ({category}):
\"\"\"{clause}\"\"\"

Rewritten Clause:
""",

    "Sentence Restructuring": """
You are an expert legal drafting assistant.

Your task is to rewrite the following contract clause by changing its sentence structure
while preserving its full legal meaning.

STRICT RULES:
- You may rearrange clauses, use passive/active voice differently, or split/merge sentences.
- Do NOT change the legal intent, obligations, rights, or any specified values.
- Maintain the clause category: {category}.
- Output ONLY the restructured clause text. No explanations or labels.

Original Clause ({category}):
\"\"\"{clause}\"\"\"

Restructured Clause:
""",

    "Vocabulary Variation": """
You are an expert legal drafting assistant.

Your task is to replace appropriate words and phrases in the following contract clause
with legally equivalent alternative expressions.

STRICT RULES:
- Only substitute synonymous legal vocabulary (e.g., "shall" → "must", "herein" → "in this Agreement").
- Do NOT change the meaning, obligations, or any numerical values.
- Maintain the clause category: {category}.
- Output ONLY the revised clause text. No explanations or labels.

Original Clause ({category}):
\"\"\"{clause}\"\"\"

Revised Clause:
""",

    "Formal Language Variation": """
You are an expert legal drafting assistant.

Your task is to rewrite the following contract clause in an alternative professional
legal drafting style.

STRICT RULES:
- Use formal legal language throughout. You may choose a more concise or more elaborate drafting style.
- Preserve ALL substantive rights, obligations, exceptions, and values exactly.
- Maintain the clause category: {category}.
- Output ONLY the rewritten clause text. No explanations or labels.

Original Clause ({category}):
\"\"\"{clause}\"\"\"

Alternative Formal Clause:
""",

    "Structural Variation": """
You are an expert legal drafting assistant.

Your task is to rewrite the following contract clause by reorganising its internal
structure — such as changing the order of conditions, using numbered sub-clauses,
or reorganising qualifying phrases — while preserving all legal obligations.

STRICT RULES:
- Do NOT add or remove any legal obligation, right, amount, date, or condition.
- Structural changes only — the meaning must remain identical.
- Maintain the clause category: {category}.
- Output ONLY the structurally varied clause text. No explanations or labels.

Original Clause ({category}):
\"\"\"{clause}\"\"\"

Structurally Varied Clause:
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo Fallback Variations (used when API key is not set)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_VARIATIONS = {
    "Confidentiality": [
        ("Legal Paraphrasing",
         "Both parties undertake to hold in strict confidence all information communicated by the other and shall refrain from disclosing such information to any external party absent prior written authorisation."),
        ("Sentence Restructuring",
         "Without the prior written consent of the disclosing party, neither party shall disclose to any third party any information received from the other under this Agreement, and each party agrees to maintain the confidentiality thereof."),
        ("Vocabulary Variation",
         "Each party covenants to preserve the secrecy of all proprietary data disclosed by the counterparty and must not divulge such data to external entities without advance written approval."),
        ("Formal Language Variation",
         "The parties hereto mutually covenant and agree that all information disclosed by one party to the other shall be held in strict confidence and shall not be revealed to any third party without the express prior written consent of the disclosing party."),
        ("Structural Variation",
         "Subject to the prior written consent of the disclosing party, all information exchanged between the parties shall remain confidential; neither party shall disclose such information to any third party under any circumstances."),
    ],
    "Termination": [
        ("Legal Paraphrasing",
         "This Agreement may be brought to an end by either party upon furnishing thirty (30) days advance written notification to the counterparty."),
        ("Sentence Restructuring",
         "Written notice of thirty (30) days must be provided by either party wishing to terminate this Agreement."),
        ("Vocabulary Variation",
         "Either contracting party may dissolve this Agreement by delivering thirty (30) days prior written notification to the other party hereto."),
        ("Formal Language Variation",
         "Notwithstanding any other provision hereof, either party shall have the right to terminate this Agreement upon thirty (30) days' written notice delivered to the other party in accordance with the notice provisions of this Agreement."),
        ("Structural Variation",
         "Upon provision of thirty (30) days' prior written notice: (i) either party may terminate this Agreement; and (ii) such termination shall be effective upon expiration of the notice period."),
    ],
    "Indemnification": [
        ("Legal Paraphrasing",
         "Each party agrees to indemnify and hold the other harmless against any claims, losses, or liabilities resulting from a breach of this Agreement by the indemnifying party."),
        ("Sentence Restructuring",
         "From any claims, damages, or liabilities arising from its breach of this Agreement, each party shall indemnify, defend, and hold the other party harmless."),
        ("Vocabulary Variation",
         "Each party shall protect, compensate, and defend the other from and against all actions, losses, and obligations arising from that party's violation of this Agreement."),
        ("Formal Language Variation",
         "Each party (the 'Indemnifying Party') hereby agrees to indemnify, defend, and hold harmless the other party (the 'Indemnified Party') from and against any and all claims, damages, costs, and liabilities arising out of the Indemnifying Party's breach of its obligations hereunder."),
        ("Structural Variation",
         "Indemnification obligations of each party shall encompass: (a) defense against third-party claims; (b) payment of resulting damages; and (c) reimbursement of reasonable legal fees, all arising from that party's breach of this Agreement."),
    ],
    "Payment Terms": [
        ("Legal Paraphrasing",
         "All invoices issued under this Agreement shall be settled within thirty (30) calendar days from the invoice date, unless the parties have agreed otherwise in writing."),
        ("Sentence Restructuring",
         "Unless otherwise mutually agreed in writing, payment for all invoices shall become due and payable within thirty (30) days of the date of issuance of the invoice."),
        ("Vocabulary Variation",
         "All billing statements rendered under this Agreement shall be remitted within thirty (30) days of the billing date, absent a contrary written arrangement between the parties."),
        ("Formal Language Variation",
         "All amounts invoiced hereunder shall be due and payable by the receiving party within thirty (30) days of the invoice date, unless a different payment schedule has been mutually agreed in writing by both parties."),
        ("Structural Variation",
         "Payment obligations: (i) all invoices shall be settled within thirty (30) days of the invoice date; (ii) any deviation from this schedule requires prior written agreement of both parties."),
    ],
    "Liability": [
        ("Legal Paraphrasing",
         "Under no circumstances shall either party bear responsibility for any indirect, incidental, special, or consequential losses arising from or in connection with this Agreement."),
        ("Sentence Restructuring",
         "Arising out of or related to this Agreement, neither party shall in any event be liable for any indirect, incidental, special, or consequential damages."),
        ("Vocabulary Variation",
         "Neither contracting party shall be accountable for any derivative, ancillary, extraordinary, or resultant losses stemming from this Agreement."),
        ("Formal Language Variation",
         "Notwithstanding any other provision of this Agreement, in no event shall either party be liable to the other for any indirect, incidental, special, exemplary, or consequential damages of any nature arising out of or related to this Agreement."),
        ("Structural Variation",
         "Excluded damages: neither party shall be liable for: (a) indirect damages; (b) incidental damages; (c) special damages; or (d) consequential damages, in each case arising out of or related to this Agreement."),
    ],
    "Intellectual Property": [
        ("Legal Paraphrasing",
         "All intellectual property rights in the deliverables produced by the Service Provider under this Agreement shall be exclusively transferred to and owned by the Client upon receipt of full payment."),
        ("Sentence Restructuring",
         "Upon full payment by the Client, all intellectual property rights in any work product created by the Service Provider under this Agreement shall vest exclusively in the Client."),
        ("Vocabulary Variation",
         "Upon complete remittance by the Client, all proprietary rights in the work output generated by the Service Provider hereunder shall be exclusively assigned to and owned by the Client."),
        ("Formal Language Variation",
         "Title to and ownership of all intellectual property rights subsisting in work product created by the Service Provider in the course of performing services hereunder shall, upon receipt of full payment therefor, vest exclusively in and be assigned to the Client."),
        ("Structural Variation",
         "Intellectual property ownership: (i) the Service Provider shall create all work product as a work-for-hire; (ii) all resulting IP rights shall transfer exclusively to the Client; and (iii) such transfer is conditional upon the Client's full payment under this Agreement."),
    ],
    "Governing Law": [
        ("Legal Paraphrasing",
         "This Agreement shall be regulated and interpreted pursuant to the laws of the State of Delaware, excluding its conflict of laws principles."),
        ("Sentence Restructuring",
         "Without regard to its conflict of law provisions, the laws of the State of Delaware shall govern and be used to construe this Agreement."),
        ("Vocabulary Variation",
         "The laws of Delaware shall control and regulate this Agreement, disregarding any provisions pertaining to conflicts of law."),
        ("Formal Language Variation",
         "This Agreement, and all matters arising out of or relating hereto, shall be governed by and construed in accordance with the internal laws of the State of Delaware, without reference to the conflict of laws rules of such state."),
        ("Structural Variation",
         "Governing law provisions: (a) this Agreement is governed by Delaware law; (b) Delaware conflict-of-law principles are expressly excluded; and (c) all disputes shall be resolved under Delaware substantive law."),
    ],
    "Data Protection": [
        ("Legal Paraphrasing",
         "Both parties must adhere to all relevant data privacy regulations, including the GDPR and any applicable national legislation implementing it."),
        ("Sentence Restructuring",
         "Including the General Data Protection Regulation (GDPR) and any national implementing legislation, the parties shall comply with all applicable data protection laws."),
        ("Vocabulary Variation",
         "Each party shall observe and abide by all pertinent data privacy statutes, encompassing the GDPR and corresponding domestic enactments."),
        ("Formal Language Variation",
         "Each party covenants and agrees to process all personal data in strict compliance with applicable data protection legislation, including but not limited to the General Data Protection Regulation (Regulation (EU) 2016/679) and all relevant national implementing measures."),
        ("Structural Variation",
         "Data protection obligations: (i) the parties shall comply with all applicable data protection laws; (ii) such laws include, without limitation, the GDPR; and (iii) national implementing legislation shall also apply where relevant."),
    ],
}


def get_demo_variations(category: str, num_samples: int = 5) -> list:
    """
    Return up to num_samples demo variations for the given category.
    Falls back to Confidentiality templates if category not found.
    """
    base = DEMO_VARIATIONS.get(category, DEMO_VARIATIONS["Confidentiality"])
    # Cycle if more samples requested than available
    result = []
    for i in range(num_samples):
        technique, text = base[i % len(base)]
        result.append({"technique": technique, "augmented_text": text, "is_demo": True})
    return result
