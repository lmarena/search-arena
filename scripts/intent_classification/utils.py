USER_INTENT_CATEGORIES = [
    "Text Processing",
    "Creative Generation",
    "Factual Lookup",
    "Info Synthesis",
    "Analysis",
    "Recommendation",
    "Explanation",
    "Guidance",
    "Other",
]

USER_INTENT_RUBRIC = {
    "Text Processing": "Users request linguistic tasks such as summarizing, translating, paraphrasing, or refining text.",
    "Creative Generation": "Users request original creative outputs, including fictional scenarios, satire, poetry, storytelling, or other forms of artistic language generation.",
    "Recommendation": "Users request suggestions or advice tailored to particular constraints, preferences, or specified criteria. Often implies personalization or ranking.",
    "Guidance": "Users request instructions, procedures, or practical advice intended to accomplish specific tasks, typically involving sequential steps or troubleshooting. Usually framed as “how to” or involving action.",
    "Analysis": "Users seek a reasoned judgment or breakdown of a topic, often involving comparison, evaluation, weighing different perspectives, and syntheszing material from many sources. Often open-ended.",
    "Explanation": "Users seek detailed clarifications, educational insights, or thorough elaborations aimed at better understanding a concept, process, or phenomeneon. May ask “why” or “how” something works without implying action.",
    "Info Synthesis": "Users seek a concise, aggregated summary that combines facts or perspectives from multiple sources. The focus is on integrating factual content without requiring reasoning, subjective interpretation, or personalization.",
    "Factual Lookup": "Users seek to retrieve a verifiable, precise, objective fact or specific information which can be answered in short phrases or sentences without requiring further context or subjective interpretation. No reasoning, explanation, or personalization required.",
    "Other": "Prompts that don't fit into any of the categories above. Prompts may be malformed, declarative in nature, or lacking in meaningful user intent.",
}


FEW_SHOT_EXAMPLES = [

    # Text Processing
    {"query": "Summarize/Read sky.cs.berkeley.edu", "primary_intent": "Text Processing", "secondary_intent": "Unassigned",},
    {"query": "Translate this paragraph from Spanish to English.", "primary_intent": "Text Processing", "secondary_intent": "Unassigned"},
    {"query": "Rephrase this email professionally for an investment-banking application.", "primary_intent": "Text Processing", "secondary_intent": "Unassigned"},

    # Creative Generation
    {"query": "Compose a witty review of ITV's Big Brother.", "primary_intent": "Creative Generation", "secondary_intent": "Unassigned"},
    {"query": "Create a humorous poem about SpongeBob.", "primary_intent": "Creative Generation", "secondary_intent": "Unassigned"},
    {"query": "Write a short fantasy story set on Mars.", "primary_intent": "Creative Generation", "secondary_intent": "Unassigned"},

    # Factual Lookup
    {"query": "How many planets are in our Solar System?", "primary_intent": "Factual Lookup", "secondary_intent": "Unassigned"},
    {"query": "When was the Mona Lisa painted?", "primary_intent": "Factual Lookup", "secondary_intent": "Unassigned"},

    # Info Synthesis
    {"query": "List the key functions of the U.S. Congress.", "primary_intent": "Info Synthesis", "secondary_intent": "Factual Lookup"},
    {"query": "Tell me the latest ai news today.", "primary_intent": "Info Synthesis", "secondary_intent": "Unassigned"},
    {"query": "GPT-4o.", "primary_intent": "Info Synthesis", "secondary_intent": "Explanation"},

    # Analysis
    {"query": "Analyze the pros and cons of nuclear energy.", "primary_intent": "Analysis", "secondary_intent": "Recommendation"},
    {"query": "When will we reach artificial general intelligence?", "primary_intent": "Analysis", "secondary_intent": "Factual Lookup"},
    {"query": "What were the strategic implications of the Cuban Missile Crisis?", "primary_intent": "Analysis", "secondary_intent": "Explanation"},

    # Recommendation
    {"query": "Best laptop for machine learning under $1500.", "primary_intent": "Recommendation", "secondary_intent": "Factual Lookup"},
    {"query": "I like literary fiction—what books should I read?", "primary_intent": "Recommendation", "secondary_intent": "Creative Generation"},
    {"query": "What programming language should I learn for web development?", "primary_intent": "Recommendation", "secondary_intent": "Guidance"},

    # Explanation
    {"query": "What led to the fall of the Roman Empire?", "primary_intent": "Explanation", "secondary_intent": "Factual Lookup"},
    {"query": "Can you explain the Heisenberg uncertainty principle?", "primary_intent": "Explanation", "secondary_intent": "Unassigned"},
    {"query": "How does gradient-descent optimization work in machine learning?", "primary_intent": "Explanation", "secondary_intent": "Guidance"},

    # Guidance
    {"query": "How do I install Docker on Ubuntu?", "primary_intent": "Guidance", "secondary_intent": "Unassigned"},
    {"query": "How do I renew my passport online?", "primary_intent": "Guidance", "secondary_intent": "Unassigned"},
    {"query": "gtts.tts.gTTSError: 429 (Too Many Requests) from TTS API. Probable cause: Unknown", "primary_intent": "Guidance", "secondary_intent": "Unassigned"},

    # Other
    {"query": "Who/How are you?", "primary_intent": "Other", "secondary_intent": "Unassigned"},
    {"query": "Talk about something please.", "primary_intent": "Other", "secondary_intent": "Unassigned"}
]

def build_augmented_prompt(query: str) -> str:
    numbered_intents = "\n".join(f"{i+1}. {cat}"
                                 for i, cat in enumerate(USER_INTENT_CATEGORIES))

    return f"""You are an impartial classifier.

TASK 1: Primary intent: choose one category that best matches the user's intent, always ask yourself, what is the user trying to get the model to do with this query. Considering them following the categories one by one in order.
TASK 2: If there is a clear secondary goal, choose one additional intent label that is also present in the query. Otherwise, return "Unassigned" if no clear second goal exists.

Allowed intent categories:
{numbered_intents}

Here are a few examples that you should follow, try to generalize beyond these examples but hold true to their semantic meaning:
{FEW_SHOT_EXAMPLES}

Respond only in valid JSON, only choosing exactly one category to fill the values:

{{
    "primary_intent": "<primary intent label>",
    "secondary_intent": "<secondary intent label>",
    "reasoning": "<your reasoning for the classification>",
}}

User query:
\"\"\"{query}\"\"\"
""".strip()