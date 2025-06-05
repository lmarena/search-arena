parse_instruction = """Task: Break down a long answer into a list of claim–citation pairs.

Specifically:
- Extract each factual or definitional claim that has one or more explicit citations like [1], [2], etc.
- For each of these, return a pair: the claim text and the associated citation(s).
- Return the result as a valid Python list of dictionaries.
- Each dictionary should contain:
  - "claim": The full, self-contained sentence(s) that states the claim. Sometimes, this may be a complete sentence or a group of sentences. Take as much context as needed to make the claim clear.
  - "citation": a list of citation strings (e.g., ["[1]"], ["[2]", "[3]"])

Example Input:
'''
1. **Runway Models**: Specialize in walking the catwalk during fashion shows[1].

2. **Supermodels**: These are highly successful fashion models known for their iconic status and influence in the fashion industry. Examples include Naomi Campbell, Cindy Crawford, and Gigi Hadid[2].

3. There are various online quizzes that match individuals with famous models based on personality traits or preferences. These quizzes are more for entertainment and self-discovery rather than professional modeling careers[2][3][4].
'''

Expected Output:
[
    {
        "claim": "Runway models specialize in walking the catwalk during fashion shows.",
        "citation": ["[1]"]
    },
    {
        "claim": "Supermodels are highly successful fashion models known for their iconic status and influence in the fashion industry.",
        "citation": ["[2]"]
    },
    {
        "claim": "Examples of supermodels include Naomi Campbell, Cindy Crawford, and Gigi Hadid.",
        "citation": ["[2]"]
    },
    {
        "claim": "Various online quizzes attempt to match individuals with famous models based on personality traits or preferences.",
        "citation": ["[2]", "[3]", "[4]"]
    },
    {
        "claim": "These model-matching quizzes are primarily designed for entertainment and self-discovery, rather than for guiding professional modeling careers.",
        "citation": ["[2]", "[3]", "[4]"]
    }
]

Now it's your turn. Return it in a json format.
"""


check_instruction = """
Task: Tell me if the following list of claims can be verified or supported by the {source}.

Specifically:
- If the claim is supported by the web content, return "support".
- If the claim is contradict to the web content, return "contradict".
- If the claim is completely irrelevant to the web content, return "irrelevant".

Claims: {claims}
Web content: {web_content}

Return in the json format:
[
    {{
        "reasoning": "<explain the reasoning>",
        "answer": "support" | "contradict" | "irrelevant"
    }},
    ...
]
"""


def count_tag_types(row):
    counts = {
        "support": 0,
        "contradict": 0,
        "irrelevant": 0,
        "total": 0
    }
    for entry in row:
        tag = entry["claim_attribution_tag"]
        if tag in counts:
            counts[tag] += 1
            counts["total"] += 1
    return counts


def generate_conv_metadata(row):
    conv_metadata = {
        "support_ratio_a": row["support_count_a"] / row["total_count_a"] if row["total_count_a"] > 0 else 0,
        "contradict_ratio_a": row["contradict_count_a"] / row["total_count_a"] if row["total_count_a"] > 0 else 0,
        "irrelevant_ratio_a": row["irrelevant_count_a"] / row["total_count_a"] if row["total_count_a"] > 0 else 0,
        "support_ratio_b": row["support_count_b"] / row["total_count_b"] if row["total_count_b"] > 0 else 0,
        "contradict_ratio_b": row["contradict_count_b"] / row["total_count_b"] if row["total_count_b"] > 0 else 0,
        "irrelevant_ratio_b": row["irrelevant_count_b"] / row["total_count_b"] if row["total_count_b"] > 0 else 0,
    }
    return conv_metadata


def reorg_data(dataset):
    """
    Reorganize the dataset to have a mapping from citation URLs to claims and metadata.
    """
    citation_entries = {}
    for i, row in dataset.iterrows():
        for side in ["resolved_a", "resolved_b"]:
            turns = row[side]
            for turn in turns:
                prompt = turn["prompt"]
                answer_chunks = turn["answer_chunks"]
                for claim, urls in answer_chunks:
                    for url in urls:
                        source = "YouTube video transcript" if "youtube" in url or "youtu.be" in url else "website"
                        if url not in citation_entries:
                            citation_entries[url] = {
                                "source": source,
                                "claims": [],
                                "metadata": []
                            }
                        citation_entries[url]["claims"].append(claim)
                        citation_entries[url]["metadata"].append({
                            "raw_message": row["messages_a"] if side == "resolved_a" else row["messages_b"],
                            "claim": claim,
                            "url": url,
                            "source": source,
                            "row_id": i,
                            "side": "model_a" if side == "resolved_a" else "model_b",
                            "prompt": prompt,
                            "model": row["model_a"] if side == "resolved_a" else row["model_b"],
                            "opponent_model": row["model_b"] if side == "resolved_a" else row["model_a"],
                            "winner": row["winner"],
                        })
    return citation_entries