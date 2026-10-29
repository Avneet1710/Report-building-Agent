## prompts.py

from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an intent classifier for a document processing assistant.

Given the user input and conversation history, classify the user's intent into one of these categories:
- qa: Questions about documents or records that do not require calculations.
    Examples:
        - "What is the policy effective date?"
        - "Who is the vendor on invoice INV-001?"
        - "Which contracts mention a termination clause?"
- summarization: Requests to summarize or extract key points from documents that do not require calculations.
    Examples:
        - "Summarize the key points of the annual report."
        - "Give me an overview of contract CON-001."
        - "What are the main terms of this insurance claim?"
- calculation: Mathematical operations or numerical computations. Or questions about documents that may require calculations
    Examples:
        - "What was the total revenue in Q3?"
        - "What is the sum of all invoices over $50,000?"
        - "Calculate 15% of the amount on INV-002."
- unknown: Cannot determine the intent clearly
    Examples:
        - "Hello"
        - "What's the weather today?"
        - "Can you help me?"

Classification rules:
- Use the conversation history to resolve references such as "it", "that one", or "the first invoice". A follow-up like "Add those up" after a document question is a calculation.
- If a question about documents needs any arithmetic (totals, averages, differences, percentages), choose calculation, not qa.
- If the request is a mix of intents, choose the one that best matches the final outcome the user wants.
- Choose unknown only when no other category reasonably fits.

Confidence scoring (a number between 0.0 and 1.0):
- 0.9 to 1.0: The request clearly matches one category and has no real ambiguity.
- 0.7 to 0.89: The request probably matches one category, but another is possible or context is needed.
- 0.4 to 0.69: The request is ambiguous and could reasonably fit two or more categories.
- 0.0 to 0.39: You are mostly guessing. In this case, use unknown.

Reasoning:
- Write one or two sentences explaining why you chose the category.
- Mention the specific words or context from the user input or history that drove the decision.
- If confidence is below 0.7, say which other category you considered and why you did not choose it.

User Input: {user_input}

Recent Conversation History:
{conversation_history}

Analyze the user's request and classify their intent with a confidence score and brief reasoning.
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

# Calculation System Prompt
# TODO: Implement the CALCULATION_SYSTEM_PROMPT. Refer to README.md Task 3.2 for details
CALCULATION_SYSTEM_PROMPT = """You are a precise calculation assistant specializing in financial and healthcare documents.

Your approach:
- Identify which document contains the data needed for the calculation
- Retrieve the document before doing any math
- Determine the exact mathematical expression the user's request requires
- Perform every calculation with the calculator tool

Guidelines:
1. First determine which document(s) you need, then search for and retrieve them using the document reader tool. Never guess or assume values that should come from a document.
2. Extract the exact numbers from the document and note which document ID each came from.
3. Work out the mathematical expression to evaluate based on the user's request (for example, a sum of line items, a percentage, or a difference between two amounts).
4. ALWAYS use the calculator tool to perform the calculation, no matter how simple it is. Never compute results yourself or do mental math, even for basic addition or multiplication.
5. If a calculation has several steps, use the calculator tool for each step.
6. In your final answer, show the expression you calculated, the result returned by the calculator tool, and the document ID(s) the numbers came from.
7. If the needed document or values cannot be found, say so clearly instead of estimating.
8. Be precise with numbers, units, currency, and dates.
"""


# TODO: Finish the function to return the correct prompt based on intent type
# Refer to README.md Task 3.1 for details
def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent.
    """
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT
    elif intent_type ==  "summarization":# TODO:  Check the intent type value
        system_prompt =  SUMMARIZATION_SYSTEM_PROMPT 
        # TODO: Set system prompt to the correct value based on intent type
    elif intent_type == "calculation": # TODO: Check the intent type value
        system_prompt =  CALCULATION_SYSTEM_PROMPT
        # TODO: Set system prompt to the correct value based on intent type
    else:
        system_prompt = QA_SYSTEM_PROMPT  # Default fallback

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""
