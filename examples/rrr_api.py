"""Build an editable RRR API example; call the model only with --execute.

Uses a tiny demonstration corpus, not an official benchmark. Credentials stay
in AIL_API_KEY. Provider-specific options belong in the saved backend config.
"""
import argparse
import json
import os

from agentinputlab import RunConfig
from agentinputlab.contracts.serialization import content_hash


def configuration():
    rewrite = "Rewrite the question as a search query: {query}"
    reader = "Answer {query} using these documents: {background}"
    return {
        "release_id": "rewrite.rrr.common_support_prompt",
        "seed": 17,
        "parameters": {"config": {
            "rewrite_prompt": rewrite, "reader_prompt": reader,
            "prompt_source_sha256": content_hash([rewrite, reader]),
            "rewrite_prompt_sha256": content_hash(rewrite),
            "reader_prompt_sha256": content_hash(reader),
            "temperature": 0.0, "max_rewriter_tokens": 512,
            "max_reader_tokens": 512, "top_k": 1,
        }},
        "budget": {"max_output_tokens": 1024, "max_model_calls": 2,
                   "max_retrieval_calls": 1},
        "backend": {
            "model_id": os.environ["AIL_MODEL"],
            "base_url": os.environ["AIL_BASE_URL"],
            "deployment_identity": {"revision": os.environ["AIL_MODEL_REVISION"]},
            "api_key_env": "AIL_API_KEY",
        },
        "retriever": {"kind": "bm25", "k1": 1.2, "b": 0.75,
                      "documents": [{"doc_id": "france",
                                     "text": "Paris is the capital of France."}]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Make real model requests")
    parser.add_argument("--query", default="What is the capital of France?")
    args = parser.parse_args()
    data = configuration()
    if args.execute:
        result = RunConfig.from_dict(data).run(args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
