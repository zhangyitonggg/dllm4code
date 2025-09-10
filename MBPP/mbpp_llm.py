import argparse
import json
import logging
import time
import os
import copy
import signal
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from datetime import datetime

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code generation configuration class, consistent with LiveCodeBench"""
    temperature: float = 0.1       # Generation temperature
    max_tokens: int = 1024          # Maximum number of generated tokens (corresponds to max_new_tokens)
    top_p: float = 0.95             # Top-p parameter
    top_k: int = 50                 # Top-k parameter
    do_sample: bool = True          # Whether to use sampling


@dataclass
class EvaluationConfig:
    """Evaluation configuration class, consistent with LiveCodeBench"""
    model_path: str = "/xxx/xxx/codellama/CodeLlama-7b-Instruct-hf"  # Model path
    device: str = "do_not_modify_cuda_here"          # Single GPU configuration (default: cuda:0)
    k: int = 1                     # k value in pass@k
    output_dir: str = "mbpp_local_results"  # Output directory
    max_samples: int | None = None  # Maximum number of samples
    timeout_seconds: int = 5        # Code execution timeout
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # Generation configuration


class GenerationResult(NamedTuple):
    """Result of a single generated sample, including token count and time"""
    task_id: int
    prompt: str
    generated_codes: list[str]      # Codes generated in k attempts
    token_counts: list[int]         # Number of tokens for each generation
    generation_times: list[float]   # Time taken for each generation (seconds)
    test_cases: list[str]
    test_results: list[bool]        # Result of each test
    test_passed: bool               # Whether at least one test passed (pass@k)


class EvaluationSummary(NamedTuple):
    """Evaluation summary results, including token/s speed"""
    model: str
    total_samples: int
    passed_samples: int
    pass_at_k: float
    total_generation_time: float    # Total generation time (seconds)
    total_tokens_generated: int     # Total number of generated tokens
    average_speed: float            # Average speed (tokens/second)
    average_time_per_sample: float  # Average time per sample


# ================================ Custom Exceptions ================================

class MBPPEvaluationError(Exception):
    """Base exception class for MBPP evaluation"""
    pass


class ModelLoadError(MBPPEvaluationError):
    """Exception raised when model loading fails"""
    pass


class DatasetLoadError(MBPPEvaluationError):
    """Exception raised when dataset loading fails"""
    pass


class CodeExecutionTimeoutError(MBPPEvaluationError):
    """Exception raised when code execution times out"""
    pass


# ================================ Helper Functions ================================

def append_to_file(filename, content):
    """Append content to a file"""
    try:
        with open(filename, 'a') as file:
            file.write(f"{content}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")


# ================================ Core Functional Classes ================================

class ModelInterface:
    """Model interface class, encapsulating model loading and code generation (consistent with LiveCodeBench style)"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the model interface"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self._eos_markers = [
            "\n```", "\nassert ", "# Example usage:",
            "\nif __name__", "\ndef main(", "\nprint("
        ]
        
        # Verify device availability
        if self.config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ModelLoadError("CUDA is not available, please check GPU configuration")
            device_id = int(self.config.device.split(":")[-1])
            if device_id >= torch.cuda.device_count():
                raise ModelLoadError(f"Specified GPU does not exist: {self.config.device}")
        
        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load model (single GPU, no dependency on accelerate)"""
        try:
            logger.info(f"Loading model to {self.config.device}: {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model (basic method, no dependency on accelerate)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.device.startswith("cuda") else torch.float32
            )
            
            # Manually move to specified device
            self.model = self.model.to(self.config.device).eval()
            logger.info(f"Model loaded successfully, using device: {self.config.device}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def generate_code(
        self,
        prompt: str,
        test_cases: list[str],
        gen_config: GenerationConfig,
    ) -> tuple[str, int, float]:
        """Generate code and return (code content, token count, time consumed)"""
        # Build prompt template (compatible with general models)
        def get_prompt():
            test_str = "\n".join([f"Test case: {t}" for t in test_cases])
            return f"""You are an expert Python programmer. Generate a Python function to solve the following problem, which must pass the given test cases.
            
Problem: {prompt}

Test cases:
{test_str}

Your solution (enclose in ```python and ```):
"""
        
        conversation = [{"role": "user", "content": get_prompt()}]
        if "Qwen3" in self.config.model_path:
            full_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,  # Add model generation prompt (e.g., "assistant:")
                enable_thinking=False
            )
        else:
            full_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True  # Add model generation prompt (e.g., "assistant:")
            )
        # print(full_prompt)
        try:
            # Encode input (remove unsupported parameters)
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]  # Remove parameters not supported by the model
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].shape[1]  # Number of input tokens
            
            # Generate code and time the process
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    do_sample=gen_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    
                )
            end_time = time.perf_counter()
            generation_time = end_time - start_time  # Time consumed for generation
            # Calculate number of generated tokens
            total_tokens = outputs.shape[1]
            generated_tokens = total_tokens - input_tokens  # Output token count - Input token count
            
            # Decode generated content
            generated_text = self.tokenizer.decode(
                outputs[0][input_tokens:],  # Only take the generated part
                skip_special_tokens=True
            )
            # print(generated_text)
            # Extract code (truncate at EOS marker)
            extracted_code = self._extract_code(generated_text)
            print(extracted_code)
            return extracted_code, generated_tokens, generation_time
            
        except Exception as e:
            error_msg = f"Failed to generate code: {e}"
            logger.error(error_msg)
            return "# Generation failed\npass", 0, 0.0

    def _extract_code(self, generated_text: str) -> str:
        """Extract code from generated text"""
        start_marker = "```python"
        end_marker = "```"
        
        # Extract code block
        start_idx = generated_text.find(start_marker)
        if start_idx == -1:
            start_marker = end_marker
            start_idx = generated_text.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = generated_text.find(end_marker, start_idx)
            code = generated_text[start_idx:end_idx].strip() if end_idx != -1 else generated_text[start_idx:].strip()
        else:
            code = generated_text.strip()
        
        # Truncate at EOS marker
        min_index = None
        for marker in self._eos_markers:
            idx = code.find(marker)
            if idx != -1 and (min_index is None or idx < min_index):
                min_index = idx
        if min_index is not None:
            code = code[:min_index].strip()
            
        return code


class CodeExecutor:
    """Code executor, tests whether generated code passes test cases"""

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    def test_code(self, code: str, test_cases: list[str]) -> bool:
        """Execute code and test cases with timeout setting"""
        def timeout_handler(signum, frame):
            raise CodeExecutionTimeoutError(f"Execution timed out ({self.timeout_seconds} seconds)")

        try:
            # Set timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            # Execute code and test cases
            namespace = {}
            try:
                # Execute generated code
                exec(code, namespace)
                # Execute all test cases
                for test in test_cases:
                    exec(test, namespace)
                return True  # All tests passed
            finally:
                # Reset signal
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except CodeExecutionTimeoutError:
            logger.debug("Code execution timed out")
            return False
        except Exception as e:
            logger.debug(f"Test failed: {e}")
            return False


class DatasetLoader:
    """Dataset loader (consistent with LiveCodeBench style)"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_mbpp_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """Load MBPP dataset (with caching)"""
        try:
            dataset = load_dataset("google-research-datasets/mbpp", split="train")
            logger.info(f"Original MBPP dataset loaded successfully, total {len(dataset)} samples")

            # Limit maximum number of samples
            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                logger.info(f"Limited number of samples to: {max_samples}")

            return dataset

        except Exception as e:
            error_msg = f"Failed to load MBPP dataset: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class MBPPEvaluator:
    """MBPP evaluator main class (consistent with LiveCodeBench style)"""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.model_interface = ModelInterface(config)
        self.code_executor = CodeExecutor(config.timeout_seconds)
        self.dataset_loader = DatasetLoader()

    def evaluate(self) -> EvaluationSummary:
        """Execute full evaluation process"""
        try:
            # Load dataset
            dataset = self.dataset_loader.load_mbpp_dataset(self.config.max_samples)
            logger.info(f"Dataset loaded successfully, total {len(dataset)} samples")

            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # Initialize statistical variables
            results = []
            total_passed = 0
            total_time = 0.0
            total_tokens = 0

            # Iterate through samples to generate code and test
            logger.info(f"Starting evaluation (pass@{self.config.k})...")
            for i, sample in enumerate(tqdm(dataset, desc="Evaluation Progress")):
                result = self._evaluate_single_sample(sample)
                results.append(result)
                
                # Accumulate statistics
                if result.test_passed:
                    total_passed += 1
                print(total_passed / (i + 1))
                total_time += sum(result.generation_times)
                total_tokens += sum(result.token_counts)

            # Calculate summary information
            pass_rate = total_passed / len(results) if results else 0.0
            avg_speed = total_tokens / total_time if total_time > 0 else 0.0
            avg_time_per_sample = total_time / len(results) if results else 0.0

            # Generate summary
            summary = EvaluationSummary(
                model=os.path.basename(self.config.model_path),
                total_samples=len(results),
                passed_samples=total_passed,
                pass_at_k=pass_rate,
                total_generation_time=total_time,
                total_tokens_generated=total_tokens,
                average_speed=avg_speed,
                average_time_per_sample=avg_time_per_sample
            )

            # Save results
            self._save_results(results, summary, output_path)
            
            # Print summary
            self._print_summary(summary, output_path)
            
            return summary

        except Exception as e:
            if isinstance(e, MBPPEvaluationError):
                raise
            raise MBPPEvaluationError(f"Error during evaluation: {e}") from e

    def _evaluate_single_sample(self, sample: dict) -> GenerationResult:
        """Evaluate a single sample (k generations + tests)"""
        generated_codes = []
        token_counts = []
        generation_times = []
        test_results = []
        for _ in range(self.config.k):
            # Generate code (get code, token count, time consumed)
            code, tokens, gen_time = self.model_interface.generate_code(
                prompt=sample["text"],
                test_cases=sample["test_list"],
                gen_config=self.config.generation_config
            )
            # code, tokens, gen_time = sample["code"], 1, 1
            # Test code
            test_passed = self.code_executor.test_code(
                code=code,
                test_cases=sample["test_list"]
            )
            
            # Record single attempt result
            generated_codes.append(code)
            token_counts.append(tokens)
            generation_times.append(gen_time)
            test_results.append(test_passed)

        # Determine if sample passed (at least one generation passed the test)
        sample_passed = any(test_results)

        return GenerationResult(
            task_id=sample["task_id"],
            prompt=sample["text"],
            generated_codes=generated_codes,
            token_counts=token_counts,
            generation_times=generation_times,
            test_cases=sample["test_list"],
            test_results=test_results,
            test_passed=sample_passed
        )

    def _save_results(self, results: list[GenerationResult], summary: EvaluationSummary, output_path: Path):
        """Save evaluation results"""
        # Save detailed results
        detailed_data = [
            {
                "task_id": r.task_id,
                "prompt": r.prompt,
                "generated_codes": r.generated_codes,
                "token_counts": r.token_counts,
                "generation_times": r.generation_times,
                "test_results": r.test_results,
                "test_passed": r.test_passed
            } for r in results
        ]
        
        # Save summary
        model_name = os.path.basename(self.config.model_path)
        detailed_file = output_path / f"mbpp_{model_name}_detailed.json"
        summary_file = output_path / f"mbpp_{model_name}_summary.json"
        
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary._asdict(), f, ensure_ascii=False, indent=2)
        
        # Record speed information
        speed_info = f"Total time: {summary.total_generation_time:.2f}s, Total tokens: {summary.total_tokens_generated}, Average speed: {summary.average_speed:.2f} token/s"
        append_to_file(f"{model_name}_mbpp_speed.txt", speed_info)

    def _print_summary(self, summary, output_path: Path):
        """Print evaluation summary"""
        logger.info("=" * 70)
        logger.info(f"MBPP Evaluation Completed!")
        logger.info(f"Model: {summary.model}")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Passed Samples: {summary.passed_samples} (pass@{self.config.k}: {summary.pass_at_k:.3f})")
        logger.info(f"Total Generation Time: {summary.total_generation_time:.2f} seconds")
        logger.info(f"Total Generated Tokens: {summary.total_tokens_generated}")
        logger.info(f"Average Generation Speed: {summary.average_speed:.2f} tokens/second")
        logger.info(f"Average Time Per Sample: {summary.average_time_per_sample:.2f} seconds")
        logger.info(f"Results Saved To: {output_path}")
        logger.info("=" * 70)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser (consistent with LiveCodeBench)"""
    parser = argparse.ArgumentParser(
        description="MBPP Code Generation Evaluation Tool (consistent with LiveCodeBench style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--model-path", type=str, help="Model path (overrides default value)")
    parser.add_argument("--device", type=str, default="cuda:4", help="Inference device (e.g., cuda:9 or cpu)")
    parser.add_argument("--k", type=int, default=1, help="k value in pass@k")
    parser.add_argument("--max-samples", type=int, help="Maximum number of evaluation samples")
    parser.add_argument("--timeout", type=int, default=5, help="Code execution timeout (seconds)")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--top-p", type=float, help="Top-p parameter")
    parser.add_argument("--top-k", type=int, help="Top-k parameter")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides default value)")

    return parser


def setup_logging(log_level: str) -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main function"""
    parser = create_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Generate configuration (consistent with LiveCodeBench)
    gen_config = GenerationConfig()
    if args.max_tokens:
        gen_config = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature or gen_config.temperature,
            top_p=args.top_p or gen_config.top_p,
            top_k=args.top_k or gen_config.top_k,
        )
    elif args.temperature or args.top_p or args.top_k:
        gen_config = GenerationConfig(
            temperature=args.temperature or gen_config.temperature,
            top_p=args.top_p or gen_config.top_p,
            top_k=args.top_k or gen_config.top_k,
        )

    # Evaluation configuration (consistent with LiveCodeBench)
    config = EvaluationConfig(
        k=args.k,
        max_samples=args.max_samples,
        timeout_seconds=args.timeout,
        generation_config=gen_config,
        device=args.device
    )
    if args.model_path:
        config.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir

    try:
        evaluator = MBPPEvaluator(config)
        summary = evaluator.evaluate()
        print(f"\n🎉 Evaluation Completed! pass@{summary.pass_at_k}: {summary.pass_at_k:.3f}, Average Speed: {summary.average_speed:.2f} tokens/second")
        print(f"📝 Results Saved To: {config.output_dir}")

    except MBPPEvaluationError as e:
        logger.error(f"Evaluation Failed: {e}")
        exit(1)


# Logging Configuration
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()