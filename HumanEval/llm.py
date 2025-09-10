import argparse
import json
import logging
import time
import os
import torch
import signal
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, List
from tqdm import tqdm
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code generation configuration class"""
    temperature: float = 0.1        # Generation temperature
    max_tokens: int = 1024          # Maximum number of tokens to generate
    top_p: float = 0.95             # Top-p sampling parameter
    top_k: int = 10                 # Top-k sampling parameter
    do_sample: bool = True         # Whether to use sampling for generation


@dataclass
class EvaluationConfig:
    """Evaluation configuration class"""
    model_path: str = "/xxx/xxx/codellama/CodeLlama-7b-Instruct-hf"  # Model path
    device: str = "21321321313"          # Running device (e.g., cuda:0 or cpu)
    k_list: list[int] = field(default_factory=lambda: [1, 10, 100])  # pass@k values list
    output_dir: str = "humaneval_results"  # Output directory
    max_samples: int | None = None  # Maximum number of samples to evaluate
    timeout_seconds: int = 5        # Code execution timeout in seconds
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # Generation configuration


class GenerationResult(NamedTuple):
    """Single generated sample result"""
    task_id: str
    prompt: str
    generated_code: str            # Full generated code
    completion: str                # Completion part (content of code block)
    token_count: int               # Number of tokens generated
    generation_time: float         # Time taken for generation
    test_passed: bool              # Whether the test passed


class EvaluationSummary(NamedTuple):
    """Evaluation summary"""
    model: str
    total_samples: int
    pass_at_k: dict[int, float]    # pass@k results
    total_generation_time: float   # Total generation time
    total_tokens_generated: int    # Total number of tokens generated
    average_speed: float           # Average generation speed (tokens/second)


# ================================ Custom Exceptions ================================

class EvaluationError(Exception):
    """Base exception for evaluation-related errors"""
    pass


class ModelLoadError(EvaluationError):
    """Exception raised when model loading fails"""
    pass


# ================================ Helper Functions ================================

def append_to_file(filename, content):
    """Append content to a file"""
    try:
        with open(filename, 'a') as file:
            file.write(f"{content}\n")
    except Exception as e:
        logger.error(f"Failed to write to file: {e}")


# ================================ Core Functional Classes ================================

class PromptHandler:
    """Prompt handling utility class (generates templates with test cases as needed)"""
    @staticmethod
    def get_prompt(prompt: str) -> str:
        """Generate a prompt containing problem description"""
        return f"""You are an intelligent programming assistant to produce Python algorithmic solutions. Can you complete the following Python function?
```python
{prompt}
```
"""


class CompletionExtractor:
    """Completion extraction logic (extracts content wrapped in ```python from generated results)"""
    @staticmethod
    def extract_from_code_block(generated_text: str) -> str:
        EOS_MARKERS = [
            "<|end_of_solution|>",
            "<|end_of_text|>",
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
        min_index = None
        for marker in EOS_MARKERS:
            index = generated_text.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        if min_index is not None:
            extracted_code = generated_text[:min_index]
        else:
            extracted_code = generated_text
        return extracted_code


class ModelInterface:
    """Local model interface class (Core adjustment: Process via chat template → Add prefix → Tokenize)"""
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_local_model()

    def _load_local_model(self) -> None:
        """Load local model"""
        try:
            logger.info(f"Loading local model to {self.config.device}: {self.config.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.config.device).eval()
            
            logger.info("Local model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load local model: {e}")

    def generate_code(self, prompt_prefix: str) -> tuple[str, str, int, float]:
        """
        Generate code following the required workflow:
        1. Generate base prompt with problem description
        2. Process conversation with chat_template (without tokenization)
        3. Add ```python\n + original prompt_prefix
        4. Tokenize final prompt and generate code
        5. Extract code block content as completion
        """
        # 1. Generate base prompt with problem description
        user_prompt = PromptHandler.get_prompt(prompt_prefix)
        
        # 2. Build conversation list
        conversation = [{"role": "user", "content": user_prompt}]
        
        # 3. Process conversation with tokenizer (generate text only, no tokenization)
        if "Qwen3" in self.config.model_path:
            full_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,  # Add generation prompt (e.g., "assistant:")
                enable_thinking=False
            )
        else:
            full_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True  # Add generation prompt (e.g., "assistant:")
            )
        
        # 4. Add ```python\n and original prompt_prefix here
        full_prompt += f"```python\n{prompt_prefix}\n"
        if "CodeLlama-7b-Instruct-hf" in self.config.model_path:
            full_prompt = full_prompt + "    "
        # print(full_prompt,"1111111")
        
        try:
            # 5. Tokenize the complete prompt now
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].shape[1]
            
            # 6. Generate code
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation_config.max_tokens,
                    temperature=self.config.generation_config.temperature,
                    top_p=self.config.generation_config.top_p,
                    top_k=self.config.generation_config.top_k,
                    do_sample=self.config.generation_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            generation_time = time.perf_counter() - start_time
            
            # Calculate number of generated tokens
            total_tokens = outputs.shape[1]
            generated_tokens = total_tokens - input_tokens
            
            # Decode generated content (includes full model response)
            generated_text = self.tokenizer.decode(
                outputs[0][input_tokens:],
                skip_special_tokens=False
            )
            if "CodeLlama-7b-Instruct-hf" in self.config.model_path:
                generated_text = "    " + generated_text
            
            # 7. Extract code block content as completion (no matching needed, directly extract code block)
            completion = CompletionExtractor.extract_from_code_block(generated_text)
            generated_code = generated_text  # Full generated text (including context)
            print(completion)
            
            return generated_code, completion, generated_tokens, generation_time
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return "", "", 0, 0.0


class CodeExecutor:
    """Code executor (preserves original logic)"""
    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    def test_code(self, prompt_prefix: str, completion: str) -> bool:
        """Test if the completed code is correct (concatenate prefix and completion)"""
        full_code = f"{prompt_prefix}{completion}"
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out ({self.timeout_seconds} seconds)")

        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            namespace = {}
            exec(full_code, namespace)
            return True
        except Exception:
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class DatasetLoader:
    """Dataset loader"""
    @staticmethod
    @lru_cache(maxsize=1)
    def load_humaneval(max_samples: int | None = None) -> dict:
        """Load HumanEval dataset"""
        try:
            problems = read_problems()
            logger.info(f"Loaded HumanEval dataset with {len(problems)} samples")
            
            if max_samples is not None and max_samples < len(problems):
                problem_ids = list(problems.keys())[:max_samples]
                problems = {k: problems[k] for k in problem_ids}
                logger.info(f"Limited number of samples to: {max_samples}")
                
            return problems
        except Exception as e:
            raise EvaluationError(f"Failed to load dataset: {e}")


class HumanEvalEvaluator:
    """HumanEval evaluator main class"""
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.model_interface = ModelInterface(config)
        self.executor = CodeExecutor(config.timeout_seconds)
        self.dataset = DatasetLoader.load_humaneval(config.max_samples)
        self.results = []

    def evaluate(self) -> EvaluationSummary:
        """Execute evaluation workflow"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        total_time = 0.0
        total_tokens = 0
        model_name = os.path.basename(self.config.model_path)
        
        # Generate code
        logger.info("Starting code generation...")
        for problem_id, problem in tqdm(self.dataset.items(), desc="Generation Progress"):
            prompt_prefix = problem["prompt"]
            
            # Generate code
            generated_code, completion, tokens, gen_time = self.model_interface.generate_code(prompt_prefix)
            
            # Test code
            test_passed = self.executor.test_code(prompt_prefix, completion)
            
            # Record result
            self.results.append(GenerationResult(
                task_id=problem_id,
                prompt=prompt_prefix,
                generated_code=generated_code,
                completion=completion,  # Code block content used directly as completion
                token_count=tokens,
                generation_time=gen_time,
                test_passed=test_passed
            ))
            
            total_time += gen_time
            total_tokens += tokens

        # Save generation results
        results_file = output_path / f"humaneval_{model_name}_results.jsonl"
        write_jsonl(
            str(results_file),
            [{"task_id": r.task_id, "completion": r.completion} for r in self.results]
        )
        logger.info(f"Generation results saved to: {results_file}")

        # Execute official evaluation
        logger.info(f"Starting evaluation (k={self.config.k_list})...")
        eval_results = evaluate_functional_correctness(
            str(results_file),
            k=self.config.k_list,
            n_workers=4
        )

        # Generate summary
        summary = EvaluationSummary(
            model=model_name,
            total_samples=len(self.results),
            pass_at_k=eval_results,
            total_generation_time=total_time,
            total_tokens_generated=total_tokens,
            average_speed=total_tokens / total_time if total_time > 0 else 0.0
        )

        # Save summary
        self._save_summary(summary, output_path)
        
        # Print summary
        self._print_summary(summary)
        
        return summary

    def _save_summary(self, summary: EvaluationSummary, output_path: Path):
        """Save evaluation summary"""
        summary_data = summary._asdict()
        summary_data["pass_at_k"] = {str(k): v for k, v in summary_data["pass_at_k"].items()}
        
        summary_file = output_path / f"{summary.model}_evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # Record speed information
        speed_info = f"Total Time: {summary.total_generation_time:.2f}s, Total Tokens: {summary.total_tokens_generated}, Average Speed: {summary.average_speed:.2f} token/s"
        append_to_file(f"{summary.model}_speed.txt", speed_info)

    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary"""
        logger.info("=" * 70)
        logger.info(f"HumanEval Evaluation Completed | Model: {summary.model}")
        logger.info(f"Total Samples: {summary.total_samples}")
        for k, v in summary.pass_at_k.items():
            logger.info(f"pass@{k}: {v:.3f}")
        logger.info(f"Total Generation Time: {summary.total_generation_time:.2f} seconds")
        logger.info(f"Total Tokens Generated: {summary.total_tokens_generated}")
        logger.info(f"Average Speed: {summary.average_speed:.2f} tokens/second")
        logger.info("=" * 70)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HumanEval Local Model Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Logging level")
    parser.add_argument("--model-path", type=str, help="Local model path (overrides default)")
    parser.add_argument("--device", type=str, default="cuda:7", help="Running device (e.g., cuda:0 or cpu)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100], help="List of pass@k values")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="humaneval_results", help="Output directory")
    
    return parser


def setup_logging(log_level: str):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger(__name__)

    # Initialize generation configuration
    gen_config = GenerationConfig()

    # Initialize evaluation configuration
    config = EvaluationConfig(
        k_list=args.k,
        max_samples=args.max_samples,
        generation_config=gen_config,
        device=args.device,
        output_dir=args.output_dir
    )
    if args.model_path:
        config.model_path = args.model_path

    try:
        evaluator = HumanEvalEvaluator(config)
        evaluator.evaluate()
    except EvaluationError as e:
        logger.error(f"Evaluation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()