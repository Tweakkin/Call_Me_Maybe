import json
import math
import numpy as np
from pathlib import Path
from llm_sdk import Small_LLM_Model

from src.registry import FunctionRegistry
from src.vocab import VocabManager
from src.fsm import JSONStateMachine

def build_prompt(registry: FunctionRegistry, user_question: str) -> str:
    """Builds the context window so the AI knows what functions exist."""
    prompt = "You are an AI assistant that calls functions. Here are the available functions:\n\n"
    for fn_name in registry.get_functions_name():
        desc = registry.get_description(fn_name)
        params = registry.get_parameters(fn_name)
        prompt += f"Function: {fn_name}\nDescription: {desc}\nParameters: {json.dumps(params)}\n\n"
        
    prompt += f"User Question: {user_question}\n"
    prompt += "Call the correct function in JSON format:\n"
    return prompt

def main():
    print("Initializing LLM SDK...")
    ai = Small_LLM_Model()
    
    print("Loading function registry...")
    registry = FunctionRegistry()
    registry.load("data/input/functions_definition.json")
    
    vocab = VocabManager(ai.get_path_to_vocab_file(), ai.decode)
    
    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        tests = json.load(f)
        
    results = []
    
    # We will test the first 3 prompts just to verify it works quickly
    for i, test in enumerate(tests):
        question = test.get("prompt", "")
        print(f"\n======================================")
        print(f"--- Test {i+1}/{len(tests)} ---")
        print(f"Question: {question}")
        print(f"======================================")
        
        prompt_text = build_prompt(registry, question)
        fsm = JSONStateMachine(ai, vocab, registry)
        
        max_tokens = 200
        steps = 0
        generated_json_string = ""
        
        while fsm.state != "DONE" and steps < max_tokens:
            allowed_tokens = fsm.get_allowed_tokens()
            
            encoded = ai.encode(prompt_text)[0].tolist()
            logits = ai.get_logits_from_input_ids(encoded)
            
            # --- THE CAGE ---
            for j in range(len(logits)):
                if j not in allowed_tokens:
                    logits[j] = -math.inf
                    
            highest_index = int(np.argmax(logits))
            decoded_word = ai.decode([highest_index])
            
            prompt_text += decoded_word
            generated_json_string += decoded_word
            
            print(decoded_word, end="", flush=True)
            
            fsm.commit(highest_index, decoded_word)
            steps += 1
            
        print("\n\nParsing JSON...")
        try:
            parsed = json.loads(generated_json_string.strip())
            
            # The test format usually expects the prompt back in the result
            final_obj = {
                "prompt": question,
                "name": parsed.get("name"),
                "parameters": parsed.get("parameters", {})
            }
            results.append(final_obj)
            print("SUCCESS! Valid JSON generated.")
        except Exception as e:
            print(f"FAILED to parse JSON: {e}")
            results.append({
                "prompt": question,
                "name": "fn_not_found",
                "parameters": {}
            })
            
    Path("data/output").mkdir(parents=True, exist_ok=True)
    with open("data/output/output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\nAll tests finished. Results written to data/output/output.json")

if __name__ == "__main__":
    main()
