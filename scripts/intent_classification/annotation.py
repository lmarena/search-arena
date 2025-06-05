import os
import json
import glob
import argparse
import datetime
import pandas as pd
from openai import OpenAI
from utils import build_augmented_prompt, USER_INTENT_CATEGORIES


def formalize_openai_request(request_id, prompt):
    return {
        "custom_id": f"request_{request_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string"
                            },
                            "primary_intent": {
                                "type": "string",
                                "enum": USER_INTENT_CATEGORIES,
                            },
                            "secondary_intent": {
                                "type": "string",
                                "enum": USER_INTENT_CATEGORIES + ["Unassigned"],
                            },
                        },
                        "required": ["reasoning", "primary_intent", "secondary_intent"],
                        "additionalProperties": False
                    }
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_augmented_prompt(prompt),
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
        }
    }


class OpenAIHelper:
    def __init__(self, client):
        self.client = client

    def get_file_content(self, file_id):
        file_response = self.client.files.content(file_id).content
        return file_response

    def get_batch(self, batch_id):
        return self.client.batches.retrieve(batch_id)

    def create_batch(self, file_path):
        batch_input_file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        obj = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Intent Classification"
            }
        )
        return obj


# Parse and extract reasoning/intents from the response
def extract_fields(content_str):
    try:
        parsed = json.loads(content_str)
        return pd.Series({
            "intent_cls_reasoning": parsed.get("reasoning"),
            "primary_intent": parsed.get("primary_intent"),
            "secondary_intent": parsed.get("secondary_intent")
        })
    except Exception:
        return pd.Series({"reasoning": None, "primary_intent": None, "secondary_intent": None})



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OpenAI requests.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.", default="data/search_arena_24k.jsonl")
    parser.add_argument("--batch_data_dir", type=str, help="Directory to save all intermediate JSONL file.")
    parser.add_argument("--batch_size", type=int, default=40000, help="Batch size for processing.")     # max size of 50k, setting to 40k for safety
    parser.add_argument("--output_data_dir", type=str, help="Directory to save all intermediate JSONL file.")
    parser.add_argument("--cmd", type=str, required=True, help="Command to run the Batch OpenAI API.", choices=["data_prep", "launch", "postprocess"])
    args = parser.parse_args()
    if args.cmd == "data_prep":
        # Input: search_arena_24k.jsonl
        # Output: Generate batch_x.jsonl (for OPENAI's batch API)
        assert args.batch_data_dir is not None, "Batch data directory must be specified for launch command."
        assert args.input_file is not None, "Input file must be specified for launch command."

        input_file = args.input_file
        data = pd.read_json(input_file, lines=True)
        data["prompt"] = data["messages_a"].apply(lambda x: x[0]["content"].strip())

        data['idx'] = pd.Series(range(len(data)))
        data = data.rename(columns={"idx": "request_id"})
        data['request'] = data.apply(lambda x: formalize_openai_request(x['request_id'], x['prompt']), axis=1)
        data['request'] = data['request'].apply(lambda x: json.dumps(x))
        # dump the data to a JSONL file
        if not os.path.exists(args.batch_data_dir):
            os.makedirs(args.batch_data_dir)
        # dump the data to batch_x.jsonl for every 40k rows
        for i in range(0, len(data), args.batch_size):
            batch_data = data.iloc[i:i + args.batch_size]
            batch_file = os.path.join(args.batch_data_dir, f"batch_{i // args.batch_size}.jsonl")
            with open(batch_file, "w") as f:
                for _, row in batch_data.iterrows():
                    f.write(row['request'] + "\n")
            print(f"Batch {i // args.batch_size} saved to {batch_file}")

    elif args.cmd == "launch":
        assert args.batch_data_dir is not None, "Batch data directory must be specified for launch command."
        assert args.output_data_dir is not None, "Output data directory must be specified for launch command."
        if not os.path.exists(args.batch_data_dir):
            assert False, "Batch data directory does not exist."
        if not os.path.exists(args.output_data_dir): 
            os.makedirs(args.output_data_dir)
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        openai_helper = OpenAIHelper(client)
        batch_info_data = []
        for fname in os.listdir(args.batch_data_dir):
            input_fname = os.path.join(args.batch_data_dir, fname)
            batch_obj = openai_helper.create_batch(input_fname)
            batch_info_data.append(batch_obj.id)
        datenow = datetime.datetime.now()
        batch_info_fname = os.path.join(args.output_data_dir, f"batch_info_{datenow}.json")
        with open(batch_info_fname, "w") as f:
            json.dump(batch_info_data, f)
        print("Batch info saved to:", batch_info_fname)
        print("Please wait for ~24 hours and run this script again with --postprocess")

    elif args.cmd == "postprocess":
        assert args.input_file is not None, "Input file must be specified for postprocess command."
        assert args.output_data_dir is not None, "Output data directory must be specified for postprocess command."
        if not os.path.exists(args.output_data_dir):
            assert False, "Output data directory does not exist."
        batch_info_file = glob.glob(os.path.join(args.output_data_dir, "batch_info_*.json"))
        assert len(batch_info_file) == 1
        batch_info_file = batch_info_file[0]
        with open(batch_info_file, "r") as f:
            batch_ids = json.load(f)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        openai_helper = OpenAIHelper(client)
        # Retrieve the batch
        for idx, batch_id in enumerate(batch_ids):
            obj = openai_helper.get_batch(batch_id)
            if obj.status == "completed":
                print("Batch is completed.")
                # Get the file content...

                output_file_id = obj.output_file_id
                if output_file_id is None:
                    error_file_id = obj.error_file_id
                    file_content = openai_helper.get_file_content(error_file_id)
                    print("Error file content:", file_content)
                    assert False, "Batch failed with error file."
                assert output_file_id is not None
                file_content = openai_helper.get_file_content(output_file_id)

                # Save the content to a file
                output_fname = os.path.join(args.output_data_dir, f"openai_output_batch_{idx}.jsonl")
                with open(output_fname, 'wb') as file:
                    file.write(file_content)
                print("Output file saved to:", output_fname)
                # Load the content
                results = []
                with open(output_fname, 'r') as file:
                    for line in file:
                        # Parsing the JSON string into a dict and appending to the list of results
                        json_object = json.loads(line.strip())
                        results.append(json_object)

                df = pd.DataFrame(results)
                df_extracted = df["response"].apply(lambda x: x["body"]["choices"][0]["message"]["content"])
                parsed_df = df_extracted.apply(extract_fields)
                input_df = pd.read_json(args.input_file, lines=True)
                # Merge the results with the input data
                final_df = pd.concat([input_df, parsed_df], axis=1)
                with open(os.path.join(args.output_data_dir, "final_output.jsonl"), "w") as f:
                    for _, row in final_df.iterrows():
                        f.write(row.to_json() + "\n")
                print("Final output saved to:", os.path.join(args.output_data_dir, "final_output.jsonl"))
                # Now, you can process the results as needed
                # Example: Calculate the total input and output tokens for cost estimation
                # Taking GPT-4.1 as an example 
                total_input = 0
                total_output = 0
                for result in results:
                    input_tokens = result['response']['body']['usage']['prompt_tokens']
                    output_tokens = result['response']['body']['usage']['completion_tokens']
                    total_input += input_tokens
                    total_output += output_tokens
                print("Total input tokens:", total_input)
                print("Total output tokens:", total_output)
                # Convert to millions for cost estimation
                price = (1.0 * total_input + 4.0 * total_output) / 1e6
                print("Total price:", price)
            else:
                print("Batch is not completed. Status:", obj.status)