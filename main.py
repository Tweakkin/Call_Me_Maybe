from llm_sdk import Small_LLM_Model
import numpy as np
import math

print("Loading model...")
obj = Small_LLM_Model()

# I ask the AI to to translate these characters, to Token IDs that it understands
t_open_brace  = obj.encode("{")[0].tolist()[0]
t_quote       = obj.encode('"')[0].tolist()[0]
t_name        = obj.encode("name")[0].tolist()[0]
t_colon       = obj.encode(":")[0].tolist()[0]
t_close_brace = obj.encode("}")[0].tolist()[0]
t_fn_tokens   = obj.encode("fn_add_numbers")[0].tolist()

prompt = "Question: What is 2 + 3?\nAnswer: "
print("\nStarting generation...")
print(f"Prompt: {prompt}", end="")

# I turn the State Machine ON
current_state = "EXPECT_START_BRACE"

fn_token_index = 0

while current_state != "DONE":
    
    if current_state == "EXPECT_START_BRACE":
        allowed_tokens = [t_open_brace]
        next_state = "EXPECT_QUOTE_1"
        
    elif current_state == "EXPECT_QUOTE_1":
        allowed_tokens = [t_quote]
        next_state = "EXPECT_NAME"
        
    elif current_state == "EXPECT_NAME":
        allowed_tokens = [t_name]
        next_state = "EXPECT_QUOTE_2"
        
    elif current_state == "EXPECT_QUOTE_2":
        allowed_tokens = [t_quote]
        next_state = "EXPECT_COLON"
        
    elif current_state == "EXPECT_COLON":
        allowed_tokens = [t_colon]
        next_state = "EXPECT_QUOTE_3"
        
    elif current_state == "EXPECT_QUOTE_3":
        allowed_tokens = [t_quote]
        next_state = "EXPECT_FN_NAME"
        
    elif current_state == "EXPECT_FN_NAME":
        # We only allow the SPECIFIC token we are currently on in the sequence
        allowed_tokens = [t_fn_tokens[fn_token_index]]
        
        # If there are still more tokens left in the function name...
        if fn_token_index < len(t_fn_tokens) - 1:
            next_state = "EXPECT_FN_NAME"  # Stay in this state!
        else:
            next_state = "EXPECT_QUOTE_4"  # Move on to the quote
        
    elif current_state == "EXPECT_QUOTE_4":
        allowed_tokens = [t_quote]
        next_state = "EXPECT_END_BRACE"
        
    elif current_state == "EXPECT_END_BRACE":
        allowed_tokens = [t_close_brace]
        next_state = "DONE"
        

    encoded = obj.encode(prompt)[0].tolist()
    logits = obj.get_logits_from_input_ids(encoded)
    

    for i in range(len(logits)):
        if i not in allowed_tokens:
            logits[i] = -math.inf
            
    highest_index = int(np.argmax(logits))
    decoded_word = obj.decode([highest_index])
    
    prompt = prompt + decoded_word
    print(decoded_word, end="", flush=True)
    
    if current_state == "EXPECT_FN_NAME" and next_state == "EXPECT_FN_NAME":
        fn_token_index += 1
        
    current_state = next_state

print("\n\nDone! The named state machine executed flawlessly.")
