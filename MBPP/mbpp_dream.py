import re
import argparse
import json
import logging
import signal
import time
import torch
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from contextlib import contextmanager

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code generation configuration class."""
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 50
    steps: int = 8
    alg: str = "maskgit_plus"
    alg_temp: float = 0.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration class."""
    model_path: str
    max_new_tokens: int
    k: int = 1  # k value in pass@k
    output_dir: str = "mbpp_dream_results"  # Replace dreamcoder with dream
    max_samples: int | None = None
    timeout_seconds: int = 5
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    device: str = "cuda:4"  # Single GPU configuration


class EvaluationResult(NamedTuple):
    """Result of a single evaluation sample."""
    task_id: int
    prompt: str
    generated_codes: list[str]  # Stores k generated code snippets
    test_cases: list[str]
    test_results: list[bool]  # Stores results of k test runs
    test_passed: bool  # Whether at least one run passed (pass@k)
    code_lengths: list[int]  # Stores lengths of k generated code snippets
    generation_times: list[float]  # Stores time taken for k generations


class EvaluationSummary(NamedTuple):
    """Evaluation summary results."""
    model: str
    model_type: str
    evaluation_method: str
    dataset: str
    k: int  # k value in pass@k
    total_samples: int
    passed_samples: int
    pass_at_k: float
    evaluation_time: float
    average_time_per_sample: float


# ================================ Custom Exceptions ================================

class MBPEvaluationError(Exception):
    """Base exception class for MBPP evaluation process."""
    pass


class ModelLoadError(MBPEvaluationError):
    """Exception raised when model loading fails."""
    pass


class DatasetLoadError(MBPEvaluationError):
    """Exception raised when dataset loading fails."""
    pass


class CodeGenerationError(MBPEvaluationError):
    """Exception raised when code generation fails."""
    pass


class CodeExecutionTimeoutError(MBPEvaluationError):
    """Exception raised when code execution times out."""
    pass


# ================================ Helper Functions ================================

@contextmanager
def multi_timer():
    """Multi-timer context manager for counting total execution time"""
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


def append_number_to_file(filename, number):
    """Append a number to the end of a file"""
    try:
        with open(filename, 'a') as file:
            file.write(f"{number}\n")
        print(f"Successfully appended number {number} to file {filename}")
    except Exception as e:
        print(f"Error while appending number: {e}")


# ================================ Core Functional Classes ================================

class ModelInterface:
    """Model interface class that encapsulates model loading and code generation functions."""

    def __init__(self, model_path: str, device: str) -> None:
        """Initialize the model interface."""
        self.model_path = model_path
        self.device = device
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._eos_markers = [
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\n```",
            "\nassert "
        ]
        self._load_model()

    def _load_model(self) -> None:
        """Load Dream model and tokenizer."""
        logger.info(f"Loading Dream model: {self.model_path}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Move model to specified device
            self._model = self._model.to(self.device).eval()
            logger.info(f"✅ Dream model loaded successfully, using device: {self.device}")

        except Exception as e:
            error_msg = f"Failed to load Dream model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    @property
    def model(self) -> AutoModel:
        """Get the model instance."""
        if self._model is None:
            raise ModelLoadError("Model not loaded correctly")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer instance."""
        if self._tokenizer is None:
            raise ModelLoadError("Tokenizer not loaded correctly")
        return self._tokenizer

    def generate_code(
        self,
        prompt: str,
        test: list,
        config: GenerationConfig,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate code (Dream-specific method)."""
        # Build prompt template
        prompt_template = f"""<|im_start|>user
You are an expert Python programmer, and here is your task: {prompt} Your code should pass the tests like this:\n\n{test}
<|im_end|>
<|im_start|>assistant
```python
"""   
        # Tokenize
        inputs = self.tokenizer(
            prompt_template,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # Generate code
        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=config.steps,
                temperature=config.temperature,
                top_p=config.top_p,
                alg=config.alg,
                alg_temp=config.alg_temp,
            )
        
        # Decode generated content (exclude input part)
        generated_text = self.tokenizer.decode(
            output.sequences[0][len(input_ids[0]):].tolist(),
            skip_special_tokens=False
        )
        
        # Find the earliest EOS marker and truncate
        min_index = None
        for marker in self._eos_markers:
            index = generated_text.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        
        # Truncate the result
        if min_index is not None:
            extracted_code = generated_text[:min_index]
        else:
            extracted_code = generated_text  # Use full content if no EOS marker is found
        # print(extracted_code)
        return extracted_code

    @staticmethod
    def _clean_generated_code(generated_text: str) -> str:
        """Extract and clean code from generated text."""
        code = generated_text.strip()
        # Clean empty lines/comments/print statements
        lines = code.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            stripped_line = line.strip()
            if not any(stripped_line.startswith(prefix) for prefix in ["print", "#"]):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


class CodeExecutor:
    """Code executor class responsible for safely executing and testing generated code."""

    def __init__(self, timeout_seconds: int = 5) -> None:
        """Initialize the code executor."""
        self.timeout_seconds = timeout_seconds

    def test_code_execution(self, code: str, test_cases: list[str]) -> bool:
        """Execute code tests with timeout to avoid infinite loops."""

        def timeout_handler(signum: int, frame: Any) -> None:
            raise CodeExecutionTimeoutError(f"Code execution timed out ({self.timeout_seconds} seconds)")

        try:
            namespace: dict[str, Any] = {}

            # Set up timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                # Execute code
                exec(code, namespace)

                # Execute test cases
                for i, test_case in enumerate(test_cases):
                    try:
                        exec(test_case, namespace)
                        logger.debug(f"Test case {i+1} passed")
                    except Exception as e:
                        logger.debug(f"Test case {i+1} failed: {e}")
                        logger.debug(f"Test case content: {test_case}")
                        return False

                return True

            finally:
                # Restore original signal handler and cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except CodeExecutionTimeoutError:
            logger.debug(f"Code execution timed out ({self.timeout_seconds} seconds), potential infinite loop")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error during test execution: {e}")
            return False


class DatasetLoader:
    """Dataset loader class."""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_mbpp_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """Load MBPP dataset (with caching)."""
        try:
            dataset = load_dataset("google-research-datasets/mbpp", split="test")
            logger.info(f"✅ MBPP dataset loaded successfully, total {len(dataset)} samples")

            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                logger.info(f"Limited number of evaluation samples to: {max_samples}")

            return dataset

        except Exception as e:
            error_msg = f"Failed to load MBPP dataset: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class MBPPEvaluator:
    """MBPP evaluator main class."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the evaluator."""
        self.config = config
        self.model_interface = ModelInterface(config.model_path, config.device)
        self.code_executor = CodeExecutor(config.timeout_seconds)
        self.dataset_loader = DatasetLoader()

    def evaluate(self) -> EvaluationSummary:
        """Execute full MBPP evaluation."""
        try:
            # Load dataset
            dataset = self.dataset_loader.load_mbpp_dataset(self.config.max_samples)

            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # Start evaluation
            logger.info(f"Starting MBPP Dream model 0-shot evaluation (pass@{self.config.k})...")  # Replace dreamcoder with dream
            results: list[EvaluationResult] = []
            total_correct = 0
            start_time = time.time()
            
            # Initialize timer
            with multi_timer() as timer:
                for i, sample in enumerate(tqdm(dataset, desc="MBPP Evaluation")):
                    # Time the processing of each sample
                    timer.start()
                    result = self._evaluate_single_sample(sample)
                    timer.stop()
                    
                    results.append(result)

                    if result.test_passed:
                        total_correct += 1

                    # Print progress periodically
                    if (i + 1) % 10 == 0:
                        current_pass_rate = total_correct / (i + 1)
                        logger.info(
                            f"Progress: {i+1}/{len(dataset)}, Current pass@{self.config.k}: {current_pass_rate:.3f}"
                        )
            
            # Save total time
            total_time = timer.get_total()
            # Modify output filename to match model name format
            output_file = f"xxx_xxx_Dream-v0-Instruct-7B_k{self.config.k}.jsonl"
            append_number_to_file("speed.txt", f"{total_time}{output_file}")

            # Calculate and save results
            return self._save_evaluation_results(
                results, start_time, total_correct, output_path
            )

        except Exception as e:
            if isinstance(e, MBPEvaluationError):
                raise
            error_msg = f"Unexpected error occurred during evaluation: {e}"
            logger.error(error_msg)
            raise MBPEvaluationError(error_msg) from e

    def _evaluate_single_sample(self, sample: dict[str, Any]) -> EvaluationResult:
        """Evaluate a single sample."""
        # Get function prompt
        prompt = sample['text']
        test = sample['test_list']
        # Perform k code generations and tests
        generated_codes = []
        test_results = []
        code_lengths = []
        generation_times = []

        for i in range(self.config.k):
            # Generate code
            generation_start = time.time()
            try:
                generated_content = self.model_interface.generate_code(
                    prompt, test, self.config.generation_config, self.config.max_new_tokens
                )

                # Clean code
                generated_code = self.model_interface._clean_generated_code(
                    generated_content
                )
            except Exception as e:
                logger.warning(f"Failed to generate code for task {sample['task_id']} (attempt {i+1}): {e}")
                generated_code = "# Code generation failed\npass"

            generation_time = time.time() - generation_start
            print(generated_code)
            # Test code
            test_passed = self.code_executor.test_code_execution(
                generated_code, sample["test_list"]
            )

            # Store results
            generated_codes.append(generated_code)
            test_results.append(test_passed)
            code_lengths.append(len(generated_code))
            generation_times.append(generation_time)

        # Calculate pass@k: sample passes if at least one of the k generations passes the test
        sample_passed = any(test_results)

        return EvaluationResult(
            task_id=sample["task_id"],
            prompt=prompt,
            generated_codes=generated_codes,
            test_cases=sample["test_list"],
            test_results=test_results,
            test_passed=sample_passed,
            code_lengths=code_lengths,
            generation_times=generation_times,
        )

    def _save_evaluation_results(
        self,
        results: list[EvaluationResult],
        start_time: float,
        total_correct: int,
        output_path: Path,
    ) -> EvaluationSummary:
        """Save evaluation results and return statistical information."""
        total_time = time.time() - start_time
        pass_rate = total_correct / len(results) if results else 0.0

        evaluation_summary = EvaluationSummary(
            model=self.config.model_path,
            model_type="dream",  # Replace dreamcoder with dream
            evaluation_method="0-shot",
            dataset="mbpp",
            k=self.config.k,
            total_samples=len(results),
            passed_samples=total_correct,
            pass_at_k=pass_rate,
            evaluation_time=total_time,
            average_time_per_sample=total_time / len(results) if results else 0,
        )

        # Convert to dictionary format for saving
        summary_dict = evaluation_summary._asdict()
        results_dict = [result._asdict() for result in results]

        # Save files, use modified model name for JSONL filename
        jsonl_file = output_path / f"xxx_xxx_Dream-v0-Instruct-7B_k{self.config.k}.jsonl"
        with open(jsonl_file, mode='w', encoding='utf-8') as f:
            for result in results:
                for code in result.generated_codes:
                    json.dump({
                        "task_id": result.task_id,
                        "completion": code
                    }, f)
                    f.write('\n')

        results_file = output_path / "mbpp_dream_evaluation_results.json"  # Replace dreamcoder with dream
        detailed_file = output_path / "mbpp_dream_detailed_results.json"  # Replace dreamcoder with dream

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # Print evaluation summary
        self._print_evaluation_summary(evaluation_summary, results_file, detailed_file)

        return evaluation_summary

    @staticmethod
    def _print_evaluation_summary(
        summary: EvaluationSummary,
        results_file: Path,
        detailed_file: Path,
    ) -> None:
        """Print evaluation result summary."""
        logger.info("=" * 50)
        logger.info(f"MBPP Dream Model 0-shot Evaluation Completed (pass@{summary.k})!")  # Replace dreamcoder with dream
        logger.info("=" * 50)
        logger.info(f"Model: {summary.model}")
        logger.info(f"Model Type: Dream")  # Replace dreamcoder with dream
        logger.info(f"Evaluation Method: 0-shot pass@{summary.k}")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Passed Samples: {summary.passed_samples}")
        logger.info(f"Pass@{summary.k}: {summary.pass_at_k:.3f}")
        logger.info(f"Evaluation Time: {summary.evaluation_time:.2f} seconds")
        logger.info(f"Average Time Per Sample: {summary.average_time_per_sample:.2f} seconds")
        logger.info(f"Results File: {results_file}")
        logger.info(f"Detailed Results: {detailed_file}")
        logger.info("=" * 50)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MBPP Dream Model 0-shot pass@k Evaluation Script",  # Replace dreamcoder with dream
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modify default model path to specified path
    parser.add_argument(
        "--model",
        default="/xxx/xxx/Dream-v0-Instruct-7B",
        help="Path to Dream model",  # Replace dreamcoder with dream
    )

    parser.add_argument(
        "--output-dir",
        default="mbpp_dream_results",  # Replace dreamcoder with dream
        help="Output directory",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of evaluation samples (for testing; evaluates all by default)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Code execution timeout in seconds",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k value for pass@k evaluation (default: 1)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser


def setup_logging(log_level: str) -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main function."""
    parser = create_arg_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)

    # Create evaluation configuration, force use of cuda:4
    config = EvaluationConfig(
        model_path=args.model,
        k=args.k,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_length,
        timeout_seconds=args.timeout,
        device="cuda:4"  # Force single GPU usage
    )

    try:
        # Create evaluator and run evaluation
        evaluator = MBPPEvaluator(config)
        import pdb
        # pdb.set_trace()
        results = evaluator.evaluate()

        # Print success message
        print(f"\n🎉 MBPP Dream Model 0-shot Evaluation Completed Successfully!")  # Replace dreamcoder with dream
        print(f"📊 Pass@{results.k}: {results.pass_at_k:.3f}")
        print(f"🤖 Model Type: Dream")  # Replace dreamcoder with dream
        print(f"🎯 Evaluation Method: 0-shot pass@{results.k}")
        print(f"📁 Results Saved To: {args.output_dir}")

    except MBPEvaluationError as e:
        logger.error(f"Evaluation Failed: {e}")
        print(f"\n❌ MBPP Dream Model Evaluation Failed: {e}")  # Replace dreamcoder with dream
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        print(f"\n❌ Unexpected Error Occurred: {e}")
        exit(1)


# ================================ Logging Configuration ================================

logger = logging.getLogger(__name__)


# ================================ Main Program Entry ================================

if __name__ == "__main__":
    main()