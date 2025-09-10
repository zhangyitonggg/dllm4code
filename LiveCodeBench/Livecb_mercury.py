import re
import argparse
import json
import logging
import time
import os
import copy
import subprocess
import requests
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from datetime import datetime 
from contextlib import contextmanager

import datasets
from datasets import load_dataset
from tqdm import tqdm


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code Generation Configuration Class - All generation parameters are controlled here"""
    temperature: float = 0.0        # Generation temperature
    max_tokens: int = 1024          # Maximum number of tokens to generate
    top_p: float = 0.95             # Top-p parameter
    presence_penalty: float = 1.5   # Presence penalty
    frequency_penalty: float = 0.0  # Frequency penalty


@dataclass
class EvaluationConfig:
    """Evaluation Configuration Class - All evaluation parameters are controlled here"""
    mercury_api_url: str = "https://api.inceptionlabs.ai/v1/chat/completions"  # Mercury API URL
    api_key: str = "sk_xxx"  # API key
    k: int = 1                     # k value in pass@k
    output_dir: str = "LiveCodeBench_mercury_results"  # Output directory
    max_samples: int | None = None  # Maximum number of samples (None means all)
    timeout_seconds: int = 10      # API timeout duration (seconds)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # Generation configuration
    custom_evaluator_path: str = "/xxx/xxx/LiveCodeBench/lcb_runner"  # Evaluator path
    run_evaluator: bool = True     # Whether to run the custom evaluator automatically


class GenerationResult(NamedTuple):
    """Result of a single generated sample"""
    question_id: str
    code_list: list[str]  # Stores k generated code snippets


class EvaluationSummary(NamedTuple):
    """Evaluation summary results"""
    model: str
    model_type: str
    generation_time: float
    average_time_per_sample: float
    total_samples: int


# ================================ Custom Exceptions ================================

class LiveCodeBenchEvaluationError(Exception):
    """Base exception class for LiveCodeBench evaluation process"""
    pass


class APICallError(LiveCodeBenchEvaluationError):
    """Exception raised when API call fails"""
    pass


class DatasetLoadError(LiveCodeBenchEvaluationError):
    """Exception raised when dataset loading fails"""
    pass


class CodeGenerationError(LiveCodeBenchEvaluationError):
    """Exception raised when code generation fails"""
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

class MercuryAPIInterface:
    """Mercury API Interface Class - Encapsulates API calls and code generation functions"""

    def __init__(self, api_url: str, api_key: str, timeout: int) -> None:
        """Initialize API interface"""
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        # List of EOS markers
        self._eos_markers = [
            "\n```",
            "\nassert ",
            "# Example usage:"
        ]

    def generate_code(
        self,
        question: dict,
        config: GenerationConfig,
    ) -> str:
        """Call Mercury API to generate code"""
        # Build prompt template - adapted to Mercury format
        def get_question_template(question):
            if question["starter_code"]:
                inner = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.```python  ```. You might only need to fill in the given function and return the given list of number. Here is the starter code:" + "\n" + question["starter_code"]
            else:
                inner = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n"
            
            prompt_template = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.
Question: {question["question_content"]}
{inner}"""
            return prompt_template
        
        prompt = get_question_template(question)
        
        # Build API request parameters
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "mercury-coder",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "top_p": config.top_p
        }
        
        try:
            # Call API
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()  # Raise HTTP errors
            generated_text = response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            error_msg = f"Mercury API call failed: {e}"
            logger.error(error_msg)
            raise APICallError(error_msg) from e
        
        # Extract code part
        return self._extract_code(generated_text, question)
    
    def _extract_code(self, generated_text: str, question: dict) -> str:
        """Extract code part from generated text"""
        # Find code block markers
        start_marker = "```python"
        end_marker = "```"
        
        start_idx = generated_text.find(start_marker)
        if start_idx != -1:
            # Skip the start marker
            start_idx += len(start_marker)
            end_idx = generated_text.find(end_marker, start_idx)
            
            if end_idx != -1:
                generated_code = generated_text[start_idx:end_idx].strip()
            else:
                generated_code = generated_text[start_idx:].strip()
        else:
            # If no code block marker is found, use the entire generated text
            generated_code = generated_text.strip()
        
        # Find the earliest EOS marker and truncate
        eos_markers = copy.deepcopy(self._eos_markers)
        if question["starter_code"]:
            eos_markers.extend(["\nif __name__","\ndef main(","\nprint("])
            
        min_index = None
        for marker in eos_markers:
            index = generated_code.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        
        # Truncate the result
        if min_index is not None:
            extracted_code = generated_code[:min_index]
        else:
            extracted_code = generated_code  # Use full content if no EOS marker is found
            
        # Clean up the extracted code
        print(extracted_code.strip())
        return extracted_code.strip()



class DatasetLoader:
    """Dataset Loader Class"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_LiveCodeBench_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """Load LiveCodeBench dataset (with caching) and filter samples after October 2024"""
        try:
            # Load original dataset
            dataset = load_dataset("/xxx/xxx/code_generation_lite", version_tag="release_v6", trust_remote_code=True)["test"]
            logger.info(f"✅ Original LiveCodeBench dataset loaded successfully, total {len(dataset)} samples")

            # Define date filtering function: only keep samples on or after October 1, 2024
            def filter_by_date(example):
                # Get contest date from the sample
                contest_date_str = example.get("contest_date")
                if not contest_date_str:  # Exclude samples without date information
                    return False
                
                try:
                    # Parse date (format example: '2023-08-21T00:00:00')
                    contest_date = datetime.strptime(contest_date_str, '%Y-%m-%dT%H:%M:%S')
                    # Set filtering threshold: October 1, 2024
                    cutoff_date = datetime(1970, 1, 1)
                    # Keep samples with date after the threshold
                    return contest_date >= cutoff_date
                except ValueError:  # Exclude samples with incorrect date format
                    logger.warning(f"Incorrect date format: {contest_date_str}, sample excluded")
                    return False

            # Apply filtering
            filtered_dataset = dataset.filter(filter_by_date)
            logger.info(f"Number of samples after filtering (after October 2024): {len(filtered_dataset)}")

            # Handle maximum sample limit
            if max_samples is not None:
                actual_max = min(max_samples, len(filtered_dataset))
                filtered_dataset = filtered_dataset.select(range(actual_max))
                logger.info(f"Maximum number of samples limited to: {actual_max}")

            return filtered_dataset

        except Exception as e:
            error_msg = f"Failed to load LiveCodeBench dataset: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class LiveCodeBenchGenerator:
    """LiveCodeBench Code Generator Main Class"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize generator"""
        self.config = config
        self.api_interface = MercuryAPIInterface(
            config.mercury_api_url, 
            config.api_key,
            config.timeout_seconds
        )
        self.dataset_loader = DatasetLoader()

    def generate(self) -> tuple[EvaluationSummary, str]:
        """Execute code generation and save results"""
        try:
            # Load dataset
            dataset = self.dataset_loader.load_LiveCodeBench_dataset(self.config.max_samples)
            logger.info(f"Dataset loaded successfully, total {len(dataset)} samples")

            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # Start code generation
            logger.info(f"Starting code generation with Mercury model (k={self.config.k})...")
            results: list[GenerationResult] = []
            
            # Initialize timer
            with multi_timer() as timer:
                for i, sample in enumerate(tqdm(dataset, desc="Code Generation Progress")):
                    # Time the processing of each sample
                    # if i > 10:
                    #     break
                    timer.start()
                    result = self._generate_single_sample(sample)
                    timer.stop()
                    
                    results.append(result)

                    # Print progress periodically
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{len(dataset)}")
            
            # Save total time
            total_time = timer.get_total()
            
            # Save generation results and return statistics
            output_file = self._save_generation_results(results, output_path)
            
            # Record generation speed
            speed_info = f"{total_time} {output_file}"
            append_number_to_file("mercury_speed.txt", speed_info)
            
            # Generate summary information
            summary = EvaluationSummary(
                model="mercury-coder",
                model_type="mercury",
                generation_time=total_time,
                average_time_per_sample=total_time / len(results) if results else 0,
                total_samples=len(results)
            )
            
            # Print result summary
            self._print_generation_summary(summary, output_file)
            
            # Run custom evaluator if needed
            if self.config.run_evaluator:
                self.run_custom_evaluator(output_file)
                
            return summary, output_file

        except Exception as e:
            if isinstance(e, LiveCodeBenchEvaluationError):
                raise
            error_msg = f"Unexpected error occurred during generation: {e}"
            logger.error(error_msg)
            raise LiveCodeBenchEvaluationError(error_msg) from e

    def _generate_single_sample(self, sample: dict[str, Any]) -> GenerationResult:
        """Generate k code snippets for a single sample"""
        # Generate k code snippets
        generated_codes = []
        for i in range(self.config.k):
            # Generate code
            try:
                generated_code = self.api_interface.generate_code(
                    sample, self.config.generation_config
                )

                logger.debug(f"Generated code: {generated_code}")
            except Exception as e:
                logger.warning(f"Failed to generate code for task {sample['question_id']} (attempt {i+1}): {e}")
                generated_code = "# Code generation failed\npass"

            generated_codes.append(generated_code)

        return GenerationResult(
            question_id=str(sample["question_id"]),
            code_list=generated_codes
        )

    def _save_generation_results(
        self,
        results: list[GenerationResult],
        output_path: Path,
    ) -> str:
        """Save generation results in the format required by the custom evaluator"""
        # Convert to the required JSON format
        output_data = [
            {
                "question_id": result.question_id,
                "code_list": result.code_list
            }
            for result in results
        ]
        
        # Generate filename (includes generation parameters)
        params = self.config.generation_config
        output_filename = f"output_mercury_tokens{params.max_tokens}_temp{params.temperature}_k{self.config.k}.json"
        output_file = output_path / output_filename
        
        # Save file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return str(output_file.absolute())

    def run_custom_evaluator(self, output_file: str) -> None:
        """Run LiveCodeBench's custom evaluator"""
        logger.info(f"Starting to evaluate generated code with custom evaluator...")
        
        # Build evaluation command
        evaluator_script = os.path.join(self.config.custom_evaluator_path, "runner", "custom_evaluator.py")
        command = [
            "python", "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", output_file
        ]
        
        try:
            # Run evaluator
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd="/xxx/xxx/LiveCodeBench"
            )
            
            # Output evaluation results
            logger.info("Evaluator Output:")
            logger.info(result.stdout)
            
            # Save evaluation results
            eval_result_file = f"{output_file}.eval_results.txt"
            with open(eval_result_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)
                
            logger.info(f"Evaluation results saved to: {eval_result_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluator run failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error occurred while running evaluator: {e}")

    @staticmethod
    def _print_generation_summary(
        summary: EvaluationSummary,
        output_file: str,
    ) -> None:
        """Print generation result summary"""
        logger.info("=" * 50)
        logger.info(f"Mercury Model Code Generation Completed!")
        logger.info("=" * 50)
        logger.info(f"Model: {summary.model}")
        logger.info(f"Model Type: {summary.model_type}")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Generation Time: {summary.generation_time:.2f} seconds")
        logger.info(f"Average Per Sample: {summary.average_time_per_sample:.2f} seconds")
        logger.info(f"Generation Result File: {output_file}")
        logger.info("=" * 50)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="LiveCodeBench Mercury Model Code Generator - Parameters controlled by configuration class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    
    parser.add_argument(
        "--no-evaluator",
        action="store_true",
        help="Do not run the custom evaluator automatically",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Mercury API key - if not provided, the default value in the configuration will be used",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate - overrides the default value in the configuration",
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="k value in pass@k",
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

    # Create generation configuration
    gen_config = GenerationConfig()
    if args.max_tokens:
        gen_config = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            presence_penalty=gen_config.presence_penalty,
            frequency_penalty=gen_config.frequency_penalty
        )

    # Create evaluation configuration
    config = EvaluationConfig(
        run_evaluator=not args.no_evaluator,
        k=args.k,
        generation_config=gen_config
    )
    
    # Update API key if provided
    if args.api_key:
        config.api_key = args.api_key

    try:
        # Create generator and run
        generator = LiveCodeBenchGenerator(config)
        summary, output_file = generator.generate()

        # Print success message
        print(f"\n🎉 Mercury Model Code Generation Completed Successfully!")
        print(f"📊 Number of Generated Samples: {summary.total_samples}")
        print(f"📝 Results Saved To: {output_file}")
        if config.run_evaluator:
            print(f"✅ Custom Evaluator Ran Automatically")

    except LiveCodeBenchEvaluationError as e:
        logger.error(f"Generation Failed: {e}")
        print(f"\n❌ Code Generation Failed: {e}")
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