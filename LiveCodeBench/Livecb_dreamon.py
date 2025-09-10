import re
import argparse
import json
import logging
import signal
import time
import torch
import os
import copy
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from datetime import datetime 
from contextlib import contextmanager

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code Generation Configuration Class - All generation parameters are controlled here (adapted for DreamOn)"""
    temperature: float = 0.2       # Generation temperature
    top_p: float = 0.9            # Top-p sampling parameter
    steps: int = 256              # Number of generation steps used by DreamOn
    alg: str = "entropy"          # Generation algorithm
    alg_temp: float = 0.0         # Algorithm temperature
    number_transfer_tokens: int = 1  # DreamOn-specific parameter


@dataclass
class EvaluationConfig:
    """Evaluation Configuration Class - All evaluation parameters are controlled here"""
    model_path: str = "/xxx/xxx/DreamOn-v0-7B"  # DreamOn model path
    k: int = 1                     # k value in pass@k
    output_dir: str = "LiveCodeBench_dreamon_results"  # Output directory
    max_samples: int | None = None  # Maximum number of samples (None means all)
    max_new_tokens: int = 256      # Maximum number of newly generated tokens
    timeout_seconds: int = 5       # Timeout duration (seconds)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # Generation configuration
    device: str = "cuda:4"         # Running device
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


class ModelLoadError(LiveCodeBenchEvaluationError):
    """Exception raised when model loading fails"""
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

class ModelInterface:
    """Model Interface Class - Encapsulates model loading and code generation functions (adapted for DreamOn)"""

    def __init__(self, model_path: str, device: str) -> None:
        """Initialize model interface"""
        self.model_path = model_path
        self.device = device
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        # DreamOn-specific EOS markers
        self._eos_markers = [
            "<|mask|>",
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\n```",
            "\nassert ",
            "!!!",
            "<|im_end|>",
            "<|dlm_pad|>",
        ]
        self._load_model()

    def _load_model(self) -> None:
        """Load DreamOn model and tokenizer"""
        logger.info(f"Loading DreamOn model: {self.model_path}")

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
            
            # Move to the specified device
            self._model = self._model.to(self.device).eval()
            logger.info(f"✅ DreamOn model loaded successfully, using device: {self.device}")

        except Exception as e:
            error_msg = f"Failed to load DreamOn model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    @property
    def model(self) -> AutoModel:
        """Get model instance"""
        if self._model is None:
            raise ModelLoadError("Model not loaded correctly")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get tokenizer instance"""
        if self._tokenizer is None:
            raise ModelLoadError("Tokenizer not loaded correctly")
        return self._tokenizer

    def generate_code(
        self,
        question:dict,
        config: GenerationConfig,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate code (DreamOn-specific method)"""
        # Build prompt template, referencing DreamCoder's template but adapted for DreamOn
        def get_question_template_answer(question):
            if question["starter_code"]:
                inner = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.```python  ```. You might only need to fill in the given function and return the given list of number. Here is the starter code:" + "\n" + question["starter_code"]
            else:
                inner = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n"
            prompt_template = f"""<|im_start|>system
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.<|im_end|>
<|im_start|>user
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {question["question_content"]}
"""+inner+f"""<|im_end|>
<|im_start|>assistant
```python
{question["starter_code"]}
"""
            return prompt_template
        prompt_template = get_question_template_answer(question)
        
        # Tokenize using DreamOn's processing method
        inputs = self.tokenizer(
            prompt_template,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs.input_ids
        input_ids_len = len(input_ids[0])
        
        # DreamOn-specific mask processing
        part2 = torch.tensor([[self.tokenizer.mask_token_id] * max_new_tokens])
        input_ids = torch.cat((input_ids, part2), dim=-1)
        input_ids = input_ids.to(device=self.device)
        
        # Generate code using DreamOn's diffusion_generate parameters
        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                max_new_tokens=max_new_tokens*2,
                output_history=False,
                steps=config.steps,
                temperature=config.temperature,
                top_p=config.top_p,
                alg=config.alg,
                alg_temp=config.alg_temp,
                number_transfer_tokens=config.number_transfer_tokens,
                return_dict_in_generate=True,
            )
        
        # Decode generated content (exclude input part)
        generated_text = self.tokenizer.decode(
            output.sequences[0][input_ids_len:].tolist(),
            skip_special_tokens=False
        )
        
        # Find the earliest EOS marker and truncate, using DreamOn's EOS markers
        eos_markers = copy.deepcopy(self._eos_markers)
        if question["starter_code"]:
            eos_markers.extend(["\nif __name__","\ndef main(","\nprint("])
        min_index = None
        for marker in eos_markers:
            index = generated_text.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        
        # Truncate the result
        if min_index is not None:
            extracted_code = generated_text[:min_index]
        else:
            extracted_code = generated_text  # If no EOS marker is found, use the entire content
            
        # Combine into complete code
        full_code = question["starter_code"] + "\n" + extracted_code
        print(full_code)
        return full_code


class DatasetLoader:
    """Dataset Loader Class - Same as the DreamCoder version"""

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
                # Get contest date from sample
                contest_date_str = example.get("contest_date")
                if not contest_date_str:  # Exclude samples without date information
                    return False
                
                try:
                    # Parse date (format: '2023-08-21T00:00:00')
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
    """LiveCodeBench Code Generator Main Class - Adapted for DreamOn"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize generator"""
        self.config = config
        self.model_interface = ModelInterface(config.model_path, config.device)
        self.dataset_loader = DatasetLoader()

    def generate(self) -> tuple[EvaluationSummary, str]:
        """Execute code generation and save results"""
        try:
            # Load dataset
            dataset = self.dataset_loader.load_LiveCodeBench_dataset(self.config.max_samples)
            
            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # Start code generation
            logger.info(f"Starting code generation with DreamOn model (k={self.config.k})...")
            results: list[GenerationResult] = []
            
            # Initialize timer
            with multi_timer() as timer:
                for i, sample in enumerate(tqdm(dataset, desc="Code Generation Progress")):
                    # Time the processing of each sample
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
            output_file = self._save_generation_results(results, total_time, output_path)
            
            # Record generation speed
            append_number_to_file("dreamon_speed.txt", f"{total_time} {output_file}")
            
            # Generate summary information
            summary = EvaluationSummary(
                model=self.config.model_path,
                model_type="dreamon",
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
                generated_content = self.model_interface.generate_code(
                    sample, self.config.generation_config, self.config.max_new_tokens
                )

                # Code cleaning
                generated_code = generated_content
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
        total_time: float,
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
        output_filename = f"output_dreamOn_t{params.temperature}_p{params.top_p}_k{self.config.k}.json"
        output_file = output_path / output_filename
        
        # Save file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return str(output_file)

    def run_custom_evaluator(self, output_file: str) -> None:
        """Run LiveCodeBench's custom evaluator"""
        logger.info(f"Starting to evaluate generated code with custom evaluator...")
        
        # Build evaluation command
        evaluator_script = os.path.join(self.config.custom_evaluator_path, "runner", "custom_evaluator.py")
        command = [
            "python", "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", "/xxx/xxx/workdir/Livecodebench_workdir/"+output_file
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
        logger.info(f"DreamOn Model Code Generation Completed!")
        logger.info("=" * 50)
        logger.info(f"Model: {summary.model}")
        logger.info(f"Model Type: DreamOn")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Generation Time: {summary.generation_time:.2f} seconds")
        logger.info(f"Average Per Sample: {summary.average_time_per_sample:.2f} seconds")
        logger.info(f"Generation Result File: {output_file}")
        logger.info("=" * 50)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser (only retains essential parameters)"""
    parser = argparse.ArgumentParser(
        description="LiveCodeBench DreamOn Model Code Generator - Parameters controlled by configuration class",
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

    # Create configuration instance (all parameters obtained from the configuration class)
    config = EvaluationConfig(
        # All parameters use default values from the configuration class
        # To modify parameters, specify them here directly or update the default values in the EvaluationConfig class
        run_evaluator=not args.no_evaluator  # Only this parameter comes from the command line
    )

    try:
        # Create generator and run
        generator = LiveCodeBenchGenerator(config)
        summary, output_file = generator.generate()

        # Print success message
        print(f"\n🎉 DreamOn Model Code Generation Completed Successfully!")
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