from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
from typing import List

from chatbots.llm import *
from chatbots.configs import llm_configs

parser = argparse.ArgumentParser(description="Initialize the LLM model.")
parser.add_argument(
    "--model", type=str, default="vicuna-13b", help="Model name for LLM."
)

args = parser.parse_args()

app = FastAPI()


class Prompt(BaseModel):
    input: str
    temperature: float = 0.5
    top_p: float = 1.0
    max_tokens: int = 64
    n_seqs: int = 1
    stop: List[str] = ["\n", "\n\n", "User", "Example"]


if args.model in llm_configs:
    model_name = llm_configs[args.model]["model_name"]
    port = llm_configs[args.model]["port"]
else:
    raise ValueError("the model type is not supported")

llm = LLM(model_name=model_name)

# local test
print(
    llm.generate(
        prompt="how are you today? i am",
        temperature=0.5,
        top_p=1.0,
        max_tokens=32,
        n_seqs=4,
        stop=["Example", "\n", "\n\n"],
    )
)


@app.post("/generate/")
async def generate_text(prompt: Prompt):
    try:
        generations = llm.generate(
            prompt=prompt.input,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            max_tokens=prompt.max_tokens,
            n_seqs=prompt.n_seqs,
            stop=prompt.stop,
        )
        return {"generated_text": generations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
