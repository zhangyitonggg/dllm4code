import re
import argparse
import json
import logging
import signal
import time
import os
import requests
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from contextlib import contextmanager

import datasets
from datasets import load_dataset
from tqdm import tqdm


# ================================ Configuration and Constants ================================
# Hardcoded API key
HARDCODED_API_KEY = "sk_xxx"

@dataclass(frozen=True)
class GenerationConfig:
    """Code generation configuration class (adapted for Mercury API parameters)"""
    temperature: float = 0.0
    top_p: float = 0.95
    presence_penalty: float = 1.5
    frequency_penalty: float = 0.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration class"""
    k: int = 1  # k value in pass@k
    output_dir: str = "mbpp_mercury_coder_results"
    max_samples: int | None = None
    max_new_tokens_list: list[int] = field(default_factory=lambda: [1024])
    timeout_seconds: int = 5
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


class EvaluationResult(NamedTuple):
    """Result of a single evaluation sample"""
    task_id: int
    prompt: str
    generated_codes: list[str]  # Stores codes generated in k attempts
    test_cases: list[str]
    test_results: list[bool]  # Stores results of k tests
    test_passed: bool  # Whether at least one test passed (pass@k)
    code_lengths: list[int]  # Stores lengths of codes generated in k attempts
    generation_times: list[float]  # Stores time taken for k generations


class EvaluationSummary(NamedTuple):
    """Evaluation summary results"""
    # Fields without default values first
    k: int  # k value in pass@k
    max_new_tokens: int  # Token limit for current test
    total_samples: int
    passed_samples: int
    pass_at_k: float
    evaluation_time: float
    average_time_per_sample: float
    # Fields with default values later
    model: str = "mercury-coder"
    model_type: str = "api"
    evaluation_method: str = "0-shot"
    dataset: str = "mbpp"


# ================================ Custom Exceptions ================================

class MBPPEvaluationError(Exception):
    """Base exception class for MBPP evaluation process"""
    pass


class APIConfigError(MBPPEvaluationError):
    """Exception for API configuration errors"""
    pass


class DatasetLoadError(MBPPEvaluationError):
    """Exception for dataset loading failures"""
    pass


class CodeGenerationError(MBPPEvaluationError):
    """Exception for code generation failures"""
    pass


class CodeExecutionTimeoutError(MBPPEvaluationError):
    """Exception for code execution timeouts"""
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
                raise RuntimeError("Timer has already started. Please stop the current timer first.")
            start_time = time.perf_counter()
        
        def stop(self):
            nonlocal total_time, start_time
            if start_time is None:
                raise RuntimeError("Please start the timer first.")
            end_time = time.perf_counter()
            total_time += end_time - start_time
            start_time = None
            
        def get_total(self):
            """Get the currently accumulated total time"""
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
        print(f"Error appending number: {e}")


# ================================ Core Functional Classes ================================

class MercuryAPIInterface:
    """Mercury API interface class that encapsulates API calls and code generation functions"""

    def __init__(self) -> None:
        """Initialize API interface (using hardcoded API key)"""
        self.api_url = "https://api.inceptionlabs.ai/v1/chat/completions"
        self.api_key = HARDCODED_API_KEY
        self._eos_markers = [
            "<|endoftext|>",
            "<|eot_id|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\n#",
            "\n```",  # Code block end marker
            "\nassert "
        ]
        
        if not self.api_key:
            raise APIConfigError("Valid Mercury API key not provided")

    def call_api(self, prompt: str, max_new_tokens: int, config: GenerationConfig) -> str:
        """Call Mercury API to generate code (using mercury-coder model)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "mercury-coder",  # Use the specified model name
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_new_tokens,
            "temperature": config.temperature,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "top_p": config.top_p
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # Raise HTTP errors
            print(response.json()["choices"][0]["message"]["content"])
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise CodeGenerationError(f"API call failed: {str(e)}") from e

    def generate_code(
        self,
        prompt: str,
        test_cases: list[str],
        config: GenerationConfig,
        max_new_tokens: int
    ) -> str:
        """Generate code and extract valid parts"""
        # Build a prompt template that meets requirements
        prompt_template = f"""
You are an expert Python programmer tasked with solving the following problem:
{prompt}

Your solution must pass these test cases:
{chr(10).join(test_cases)}

Respond with ONLY the Python code, starting with ```python, followed by the complete function, ending with ```.
Do not include any explanations or additional text.
"""
        
        # Call API to generate content
        generated_text = self.call_api(prompt_template, max_new_tokens, config)
        
        # Extract complete content starting from ```python
        return self.extract_code_block(generated_text)

    def extract_code_block(self, generated_text: str) -> str:
        """Extract only the content starting from ```python until EOS marker or code block end"""
        start_marker = "```python"
        start_idx = generated_text.find(start_marker)
        
        if start_idx == -1:
            # If no code block marker is found, return empty string (or raw text as fallback)
            logger.warning("```python marker not found, using full generated text")
            code_content = generated_text.strip()
        else:
            # Extract content after ```python
            code_content = generated_text[start_idx + len(start_marker):].strip()
        
        # Truncate using EOS markers (keep content up to the first EOS marker)
        min_index = None
        for marker in self._eos_markers:
            index = code_content.find(marker)
            if index != -1 and (min_index is None or index < min_index):
                min_index = index
        
        if min_index is not None:
            return code_content[:min_index].strip()
        return code_content

    @staticmethod
    def clean_generated_code(generated_text: str) -> str:
        """Clean generated code (remove print statements and comments)"""
        lines = generated_text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not any(stripped_line.startswith(prefix) for prefix in ["print", "#"]):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


class CodeExecutor:
    """Code executor class responsible for safely executing and testing generated code"""

    def __init__(self, timeout_seconds: int = 5) -> None:
        """Initialize code executor"""
        self.timeout_seconds = timeout_seconds

    def test_code_execution(self, code: str, test_cases: list[str]) -> bool:
        """Execute code tests with timeout to avoid infinite loops"""

        def timeout_handler(signum: int, frame: Any) -> None:
            raise CodeExecutionTimeoutError(f"Code execution timed out ({self.timeout_seconds} seconds)")

        try:
            namespace: dict[str, Any] = {}

            # Set up timeout handling
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                # Execute code
                exec(code, namespace)

                # Execute test cases
                for test_case in test_cases:
                    try:
                        exec(test_case, namespace)
                    except Exception as e:
                        logger.debug(f"Test case execution failed: {e}")
                        return False

                return True

            finally:
                # Restore original signal handler and cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except CodeExecutionTimeoutError:
            logger.debug(f"Code execution timed out ({self.timeout_seconds} seconds)")
            return False
        except Exception as e:
            logger.debug(f"Test execution error: {e}")
            return False


class DatasetLoader:
    """Dataset loader class"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_mbpp_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """Load MBPP dataset (with caching)"""
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
    """MBPP evaluator main class"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize evaluator"""
        self.config = config
        self.api_interface = MercuryAPIInterface()  # No need to pass API key, use hardcoded value
        self.code_executor = CodeExecutor(config.timeout_seconds)
        self.dataset_loader = DatasetLoader()

    def evaluate(self) -> list[EvaluationSummary]:
        """Execute complete MBPP evaluation (multiple max_new_tokens tests)"""
        summaries = []
        dataset = self.dataset_loader.load_mbpp_dataset(self.config.max_samples)
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)

        # Test each max_new_tokens value
        for max_new_tokens in self.config.max_new_tokens_list:
            logger.info(f"\n===== Starting test with max_new_tokens = {max_new_tokens} =====")
            results: list[EvaluationResult] = []
            total_correct = 0
            start_time = time.time()
            
            # Initialize timer
            with multi_timer() as timer:
                for sample in tqdm(dataset, desc=f"MBPP Evaluation (max_tokens={max_new_tokens})"):
                    timer.start()
                    result = self._evaluate_single_sample(sample, max_new_tokens)
                    timer.stop()
                    results.append(result)
                    
                    if result.test_passed:
                        total_correct += 1

            # Save total time
            total_time = timer.get_total()
            output_file = f"mbpp_mercury_coder_k{self.config.k}_tokens{max_new_tokens}.jsonl"
            append_number_to_file("speed.txt", f"{total_time} {output_file}")

            # Save evaluation results for current parameters
            summary = self._save_evaluation_results(
                results, start_time, total_correct, output_path, max_new_tokens
            )
            summaries.append(summary)

        return summaries

    def _evaluate_single_sample(self, sample: dict[str, Any], max_new_tokens: int) -> EvaluationResult:
        """Evaluate a single sample"""
        prompt = sample['text']
        test_cases = sample['test_list']
        
        generated_codes = []
        test_results = []
        code_lengths = []
        generation_times = []

        for _ in range(self.config.k):
            # Generate code
            generation_start = time.time()
            try:
                # Call API to generate code and extract code block
                raw_code = self.api_interface.generate_code(
                    prompt, test_cases, self.config.generation_config, max_new_tokens
                )
                # Clean code
                cleaned_code = self.api_interface.clean_generated_code(raw_code)
            except Exception as e:
                logger.warning(f"Failed to generate code for task {sample['task_id']}: {e}")
                cleaned_code = "# Code generation failed\npass"

            generation_time = time.time() - generation_start
            
            # Test code
            test_passed = self.code_executor.test_code_execution(cleaned_code, test_cases)

            # Store results
            generated_codes.append(cleaned_code)
            test_results.append(test_passed)
            code_lengths.append(len(cleaned_code))
            generation_times.append(generation_time)

        # Calculate pass@k
        sample_passed = any(test_results)

        return EvaluationResult(
            task_id=sample["task_id"],
            prompt=prompt,
            generated_codes=generated_codes,
            test_cases=test_cases,
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
        max_new_tokens: int
    ) -> EvaluationSummary:
        """Save evaluation results and return statistical information"""
        total_time = time.time() - start_time
        pass_rate = total_correct / len(results) if results else 0.0

        evaluation_summary = EvaluationSummary(
            k=self.config.k,
            max_new_tokens=max_new_tokens,
            total_samples=len(results),
            passed_samples=total_correct,
            pass_at_k=pass_rate,
            evaluation_time=total_time,
            average_time_per_sample=total_time / len(results) if results else 0,
        )

        # Convert to dictionary format for saving
        summary_dict = evaluation_summary._asdict()
        results_dict = [result._asdict() for result in results]

        # Save generated code results in JSONL format
        jsonl_file = output_path / f"mbpp_mercury_coder_k{self.config.k}_tokens{max_new_tokens}.jsonl"
        with open(jsonl_file, mode='w', encoding='utf-8') as f:
            for result in results:
                for code in result.generated_codes:
                    json.dump({
                        "task_id": result.task_id,
                        "completion": code
                    }, f)
                    f.write('\n')

        # Save summary results
        summary_file = output_path / f"summary_k{self.config.k}_tokens{max_new_tokens}.json"
        detailed_file = output_path / f"detailed_k{self.config.k}_tokens{max_new_tokens}.json"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # Print result summary
        self._print_evaluation_summary(evaluation_summary, summary_file)

        return evaluation_summary

    @staticmethod
    def _print_evaluation_summary(
        summary: EvaluationSummary,
        summary_file: Path,
    ) -> None:
        """Print evaluation result summary"""
        logger.info("=" * 50)
        logger.info(f"MBPP Mercury Coder Evaluation Completed (pass@{summary.k}, max_tokens={summary.max_new_tokens})")
        logger.info("=" * 50)
        logger.info(f"Model: mercury-coder")
        logger.info(f"Evaluation Method: 0-shot pass@{summary.k}")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Passed Samples: {summary.passed_samples}")
        logger.info(f"Pass@{summary.k}: {summary.pass_at_k:.3f}")
        logger.info(f"Evaluation Time: {summary.evaluation_time:.2f} seconds")
        logger.info(f"Average Time Per Sample: {summary.average_time_per_sample:.2f} seconds")
        logger.info(f"Result File: {summary_file}")
        logger.info("=" * 50)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser (remove API key parameter)"""
    parser = argparse.ArgumentParser(
        description="MBPP Mercury Coder Model 0-shot pass@k Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        default="mbpp_mercury_coder_results",
        help="Output directory",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of evaluation samples (for testing; evaluates all by default)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs='+',
        default=[1024],
        help="List of max_new_tokens values for testing (space-separated)",
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
    """Configure logging settings"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main function"""
    parser = create_arg_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)

    # Create evaluation configuration (no API key parameter needed)
    config = EvaluationConfig(
        k=args.k,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens_list=args.max_tokens,
        timeout_seconds=args.timeout,
    )

    try:
        # Create evaluator and run evaluation
        evaluator = MBPPEvaluator(config)
        results = evaluator.evaluate()

        # Print final summary
        print("\n🎉 All MBPP Mercury Coder Evaluations Completed Successfully!")
        print("📊 Evaluation Result Summary:")
        for summary in results:
            print(f"  max_new_tokens={summary.max_new_tokens}, pass@{summary.k}={summary.pass_at_k:.3f}")
        print(f"📁 Results Saved To: {args.output_dir}")

    except MBPPEvaluationError as e:
        logger.error(f"Evaluation Failed: {e}")
        print(f"\n❌ MBPP Evaluation Failed: {e}")
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