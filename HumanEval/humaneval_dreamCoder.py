import torch
import json
import os
from transformers import AutoModel, AutoTokenizer
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
import time
from contextlib import contextmanager

@contextmanager
def multi_timer():
    total_time = 0.0
    start_time = None
    
    class Timer:
        def start(self):
            nonlocal start_time
            if start_time is not None:
                raise RuntimeError("Timer has already started, please stop the current timer first")
            start_time = time.perf_counter()
        
        def stop(self):
            nonlocal total_time, start_time
            if start_time is None:
                raise RuntimeError("Please start the timer first")
            end_time = time.perf_counter()
            total_time += end_time - start_time
            start_time = None
            
        def get_total(self):
            """Get the current accumulated total time"""
            return total_time
    
    try:
        yield Timer()
    finally:
        print(f"Total execution time of all timed statements: {total_time:.6f} seconds")

# Load model and tokenizer
def append_number_to_file(filename, number):
    """
    Append a number to the end of a file
    
    Parameters:
        filename (str): Target file path
        number: Number to append (supports integers, floats, and other numeric types)
    """
    try:
        # Open file in append mode, create if it doesn't exist
        with open(filename, 'a') as file:
            # Append number, with newline as needed
            file.write(f"{number}\n")  # Add newline to put each number on a separate line
        print(f"Successfully appended number {number} to file {filename}")
    except Exception as e:
        print(f"Error when appending number: {e}")

model_path = "/xxx/xxx/Dream-Coder-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda:6").eval()

# Define EOS marker list
EOS_MARKERS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#",
    "\n```",
    "\ndef",
    "\nclass ",
    "\nimport ",
    "\nfrom ",
    "\nassert "
]

# Load HumanEval dataset
problems = read_problems()

# List to store generated results
results = []

# Process each problem
with multi_timer() as timer:
    for problem_id, problem in problems.items():
        print(f"Processing problem: {problem_id}")
        
        # Extract problem prompt
        prompt = problem['prompt']
        
        # Build new prompt template
        prompt_template = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?{prompt}<|im_end|>
<|im_start|>assistant{prompt}
"""
        # Directly tokenize the template
        inputs = tokenizer(
            prompt_template,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Adjust according to model's maximum length
        )
        input_ids = inputs.input_ids.to(device="cuda:6")
        attention_mask = inputs.attention_mask.to(device="cuda:6")
        
        # Generate code
        timer.start()
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            output_history=False,
            return_dict_in_generate=True,
            steps=2048,
            temperature=0.0,
            top_p=0.95,
            alg="maskgit_plus",
            alg_temp=0.,
        )
        timer.stop()
        # Decode generated content (excluding input part)
        generated_text = tokenizer.decode(
            output.sequences[0][len(input_ids[0]):].tolist(),
            skip_special_tokens=False
        )
        
        # Find earliest occurrence of EOS marker and truncate
        min_index = None
        for marker in EOS_MARKERS:
            index = generated_text.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        
        # Truncate result
        if min_index is not None:
            extracted_code = generated_text[:min_index]
        else:
            extracted_code = generated_text  # If no EOS marker found, use entire content
        
        print(extracted_code)
        
        # Save result
        results.append({
            "task_id": problem_id,
            "completion": extracted_code
        })
    p = timer.get_total()


# Define result file path
output_file = "humaneval_dreamcoder_RQ1.jsonl"
append_number_to_file("speed.txt", f"{p}{output_file}")
# Save generated results to JSONL file
with open(output_file, mode='w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f)
        f.write('\n')

# Evaluate generated code
print("Starting evaluation...")
evaluation_results = evaluate_functional_correctness(output_file, k=[1, 10, 100], n_workers=4)

# Print evaluation results
print("\nEvaluation Results:")
for k, v in evaluation_results.items():
    print(f"k={k}: {v}")

print(f"Results saved to {output_file}")
    