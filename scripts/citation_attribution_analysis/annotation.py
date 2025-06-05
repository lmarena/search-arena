import re
import os
import json
import argparse
import pandas as pd
from enum import Enum
from tqdm import tqdm
from google import genai
from pydantic import BaseModel
from utils import parse_instruction, check_instruction, count_tag_types, generate_conv_metadata, reorg_data
tqdm.pandas()


def llm_citation_claim_parsing(text: str, web_trace: dict, model_name: str="gemini-2.5-flash-preview-04-17"):

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = parse_instruction + "\n\n" + text
    messages = [{"role": "user", "parts": [{"text": prompt}]}]

    class ClaimCitationPair(BaseModel):
        claim: str
        citation: list[int]

    for round in range(5):
        response = client.models.generate_content(
            model=model_name,
            contents=messages,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[ClaimCitationPair],
                'temperature': round * 0.05, # increate temperature if it fails
                'max_output_tokens': 64000,
            }
        )

        chunks = []
        if response is None:
            continue
        for item in response.parsed:
            if isinstance(item, ClaimCitationPair):
                citation_urls = []
                for citation in item.citation:
                    key = f"[{citation}]"
                    if key in web_trace:
                        citation_urls.append(web_trace[key])
                if len(citation_urls) > 0:
                    chunks.append((item.claim, citation_urls))
            else:
                print(f"Unexpected item type: {type(item)}")
        return chunks
    return []  # No valid response


def resolve_citations(messages, web_traces):
    """
    Resolves citations in a list of model messages.
    Each result includes:
      - 'prompt': original question
      - 'answer_chunks': list of (claim_text, list_of_urls) tuples
    """
    results = []
    N_rounds = len(messages) // 2

    for round_idx in range(N_rounds):
        question_msg = messages[round_idx * 2]
        answer_msg = messages[round_idx * 2 + 1]
        question_text = question_msg.get("content", "")
        answer_text = answer_msg.get("content", "")

        # assert question_text and answer_text, "Question and Answer messages must contain 'content'."

        # No reference or inline citation --> skip
        if len(web_traces[round_idx]) == 0 or re.search(r"\[(\d+)\]", answer_text) is None:
            continue
        web_trace_dict = {}
        for i in range(len(web_traces[round_idx])):
            marker = web_traces[round_idx][i][0]
            url = web_traces[round_idx][i][1]
            web_trace_dict[marker] = url
        chunks = llm_citation_claim_parsing(answer_text, web_trace_dict)

        results.append({
            "prompt": question_text,
            "answer_chunks": chunks
        })

    return results


def llm_misattribution_check(claims: str, web_content: str, source) -> dict:

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = check_instruction.format(claims=claims, web_content=web_content, source=source)
    messages = [{"role": "user", "parts": [{"text": prompt}]}]
    
    class AnswerType(str, Enum):
        support = "support"
        contradict = "contradict"
        irrelevant = "irrelevant"

    class GroundingResult(BaseModel):
        reasoning: str
        answer: AnswerType

    for round in range(5):
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=messages,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[GroundingResult],
                'temperature': round * 0.05,
                'max_output_tokens': 64000,
            }
        )
        if response is None:
            continue
        if len(response.parsed) != len(claims):
            print(f"Warning: response length {len(response.parsed)} does not match claims length {len(claims)}")
            continue
        return {
            "claims": claims,
            "attribute_label": [item.answer for item in response.parsed],
            "attribute_reasoning": [item.reasoning for item in response.parsed]
        }

    return {
        "claims": claims,
        "attribute_label": ["Failed"] * len(claims),
        "attribute_reasoning": ["Failed"] * len(claims)
    }


def append_record(output_path: str, record: dict) -> None:
    with open(output_path, "a", encoding="utf‑8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Step 0: Prepare the input files (url_fname and dataset_fname)
    # The `url_fname` should be a JSONL file containing scraped URLs and their content.
    # The `dataset_fname` should be a JSONL file containing the subsampled dataset.
    # In `url_fname`, each line in the file should be a JSON object with "url" and "text" fields.
    # Example: {"url": "https://example.com", "text": "Example content from the URL."}
    parser = argparse.ArgumentParser(description="Check citation attribution with LLM.")
    parser.add_argument("--url_fname", type=str, default="data/scraped_urls.jsonl", help="Path to the scraped URLs JSONL file.")
    parser.add_argument("--dataset_fname", type=str, default="data/subset.jsonl", help="Path to the dataset JSONL file.")
    parser.add_argument("--intermediate_fname", type=str, default="data/resolved_citations.jsonl", help="Path to the intermediate JSONL file.")
    parser.add_argument("--llm_judge_fname", type=str, default="data/judged_citations.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("--output_fname", type=str, default="data/citation_attribution_data.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("--cmd", type=str, required=True, help="Command to run the annotation script.", choices=["citation_resolve", "citation_check"])
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    if args.cmd == "citation_resolve":
        # Step 1: Sanity check and resolve citations with LLM
        url_df = pd.read_json(args.url_fname, lines=True)
        url_dict = dict(zip(url_df["url"], url_df["text"]))
        dataset = pd.read_json(args.dataset_fname, lines=True)

        dataset = dataset[(dataset["primary_intent"] != "Other")]
        assert not (dataset["primary_intent"] == "Other").any(), "There are still rows with 'Other' as primary_intent."
        
        # Extract web_search_trace from metadata
        dataset["web_trace_a"] = dataset["system_a_metadata"].apply(lambda x: x.get("web_search_trace", []))
        dataset["web_trace_b"] = dataset["system_b_metadata"].apply(lambda x: x.get("web_search_trace", []))
        print("Resolving citations...")
        dataset["resolved_a"] = dataset.progress_apply(
            lambda row: resolve_citations(row["messages_a"], row["web_trace_a"]), axis=1
        )
        dataset["resolved_b"] = dataset.progress_apply(
            lambda row: resolve_citations(row["messages_b"], row["web_trace_b"]), axis=1
        )
        dataset.to_json(args.intermediate_fname, lines=True, orient="records")

    elif args.cmd == "citation_check":
        # Step 2: Do acutal citation misattribution check
        dataset = pd.read_json(args.intermediate_fname, lines=True)
        url_df = pd.read_json(args.url_fname, lines=True)
        url_dict = dict(zip(url_df["url"], url_df["text"]))
        # reorg them from claim: urls to urls: claims to save LLM calls
        citation_entries = reorg_data(dataset)
        print(len(citation_entries), "entries to check")
        for url in tqdm(citation_entries, desc="Checking Attribution"):
            entry = citation_entries[url]
            assert url in url_dict, f"URL {url} not found in url_dict"
            result = llm_misattribution_check(entry["claims"], url_dict[url], entry["source"])
            entry["result"] = result
            # The LLM script is not stable and therefore we save the results incrementally
            append_record(args.llm_judge_fname, entry)

        # Final processing
        input_df = pd.read_json(args.llm_judge_fname, lines=True)
        # count the number of claim_tags for each side for each row
        tag_counts_df_a = input_df['claim_attribution_tags_a'].apply(count_tag_types).apply(pd.Series)
        tag_counts_df_b = input_df['claim_attribution_tags_b'].apply(count_tag_types).apply(pd.Series)
        # Rename columns if desired
        tag_counts_df_a.columns = ['support_count_a', 'contradict_count_a', 'irrelevant_count_a', 'total_count_a']
        tag_counts_df_b.columns = ['support_count_b', 'contradict_count_b', 'irrelevant_count_b', 'total_count_b']

        # Merge back into original DataFrame
        input_df = pd.concat([input_df, tag_counts_df_a, tag_counts_df_b], axis=1)
        input_df["conv_metadata"] = input_df.apply(generate_conv_metadata, axis=1)
        input_df.to_json(args.output_fname, lines=True, orient="records")