import argparse
import json
import logging
import time
import os
import copy
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple
from datetime import datetime 

import datasets
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================ Configuration and Constants ================================

@dataclass(frozen=True)
class GenerationConfig:
    """Code Generation Configuration Class - Core parameters retained"""
    temperature: float = 0.1        # Generation temperature
    max_tokens: int = 1024          # Maximum number of tokens to generate
    top_p: float = 0.95             # Top-p sampling parameter
    top_k: int = 50                 # Top-k sampling parameter
    do_sample: bool = True          # Whether to use sampling


@dataclass
class EvaluationConfig:
    """Evaluation Configuration Class - Simplified Version"""
    model_path: str = "/xxx/xxx/Seed-Coder-8B-Instruct"  # Model path
    device: str = "fasfdadasd"  # Fixed to use the first GPU
    k: int = 1                     # k value in pass@k
    output_dir: str = "LiveCodeBench_local_results"  # Output directory
    max_samples: int | None = None  # Maximum number of samples
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # Generation configuration
    custom_evaluator_path: str = "/xxx/xxx/LiveCodeBench/lcb_runner"  # Evaluator path
    run_evaluator: bool = True     # Whether to run the custom evaluator automatically


class GenerationResult(NamedTuple):
    """Result of a single generated sample"""
    question_id: str
    code_list: list[str]  # Stores k generated code snippets
    token_counts: list[int]  # Number of tokens for each generation
    time_taken: list[float]  # Time taken for each generation (seconds)


class EvaluationSummary(NamedTuple):
    """Evaluation summary results"""
    model: str
    total_generation_time: float
    total_tokens_generated: int
    average_speed: float  # token/s
    total_samples: int
    average_tokens_per_sample: float


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


# ================================ Helper Functions ================================

def append_to_file(filename, content):
    """Append content to a file"""
    try:
        with open(filename, 'a') as file:
            file.write(f"{content}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")


# ================================ Core Functional Classes ================================

class LocalModelInterface:
    """Local Large Language Model Interface Class - Uses only a single GPU"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize model interface"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self._eos_markers = ["\n```", "\nassert ", "# Example usage:"]
        
        # Verify if the specified device is available
        if self.config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ModelLoadError("CUDA is not available, please check GPU configuration")
            device_id = int(self.config.device.split(":")[-1])
            if device_id >= torch.cuda.device_count():
                raise ModelLoadError(f"Specified GPU device does not exist: {self.config.device}")
        
        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load local model and tokenizer (uses only a single GPU, no dependency on accelerate)"""
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
            
            # Basic model loading parameters
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.config.device.startswith("cuda") else torch.float32
            }
            
            # Do not use device_map to avoid dependency on accelerate
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,** model_kwargs
            )
            
            # Manually move the model to the specified device
            self.model = self.model.to(self.config.device)
            
            # Ensure the model is in evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully, using device: {self.config.device}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def generate_code(
        self,
        question: dict,
        config: GenerationConfig,
    ) -> tuple[str, int, float]:
        """Generate code using the local model, returns code, token count, and time taken"""
        # Build prompt template (packaged into "user" conversation format)
        def get_prompt(question):
            if question["starter_code"]:
                inner = f"You will use the following starter code to write the solution:\n{question['starter_code']}"
            else:
                inner = "Read inputs from stdin, solve the problem, and write output to stdout.\n```python\n# YOUR CODE HERE\n```"
            
            # Build user prompt content
            user_content = f"""You are an expert Python programmer. Generate a correct Python program for the problem.
Question: {question["question_content"]}
{inner}"""
            
            # Package into conversation format with "user" key
            return [{"role": "user", "content": user_content}]
        
        # Get prompt in conversation format
        conversation = get_prompt(question)
        
        try:
            # Convert conversation format to model input text
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
            # Encode input and move to target device, exclude unnecessary parameters
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            # Remove parameters not used by the model
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            # Move to target device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Generation configuration
            generation_kwargs = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": config.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate code and record time
            start_time = time.perf_counter()
            with torch.no_grad():  # Disable gradient computation to save memory
                outputs = self.model.generate(**inputs,** generation_kwargs)
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            # Calculate number of generated tokens
            input_tokens = inputs["input_ids"].shape[1]
            total_tokens = outputs.shape[1]
            generated_tokens = total_tokens - input_tokens
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][input_tokens:], 
                skip_special_tokens=True
            )
            
        except Exception as e:
            error_msg = f"Failed to generate code: {e}"
            logger.error(error_msg)
            return "# Code generation failed\npass", 0, 0.0
        
        # Extract code part
        extracted_code = self._extract_code(generated_text, question)
        print(extracted_code)
        return extracted_code, generated_tokens, time_taken
    
    def _extract_code(self, generated_text: str, question: dict) -> str:
        """Extract code part from generated text"""
        start_marker = "```python"
        end_marker = "```"
        
        start_idx = generated_text.find(start_marker)
        if start_idx == -1:
            start_marker = end_marker
            start_idx = generated_text.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = generated_text.find(end_marker, start_idx)
            generated_code = generated_text[start_idx:end_idx].strip() if end_idx != -1 else generated_text[start_idx:].strip()
        else:
            generated_code = generated_text.strip()
        
        # Apply EOS markers
        eos_markers = copy.deepcopy(self._eos_markers)
        if question["starter_code"]:
            eos_markers.extend(["\nif __name__", "\ndef main(", "\nprint("])
            
        min_index = None
        for marker in eos_markers:
            index = generated_code.find(marker)
            if index != -1 and (min_index is None or index < min_index):
                min_index = index
        
        return generated_code[:min_index].strip() if min_index is not None else generated_code.strip()


class DatasetLoader:
    """Dataset Loader Class"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """Load LiveCodeBench dataset and filter samples after October 2024"""
        try:
            dataset = load_dataset(
                "/xxx/xxx/code_generation_lite", 
                version_tag="release_v6", 
                trust_remote_code=True
            )["test"]
            logger.info(f"Original dataset loaded successfully, total {len(dataset)} samples")

            # Filter samples after October 2024
            def filter_by_date(example):
                contest_date_str = example.get("contest_date")
                if not contest_date_str:
                    return False
                try:
                    contest_date = datetime.strptime(contest_date_str, '%Y-%m-%dT%H:%M:%S')
                    return contest_date >= datetime(1970, 1, 1)
                except ValueError:
                    return False

            filtered_dataset = dataset.filter(filter_by_date)
            logger.info(f"Number of samples after filtering: {len(filtered_dataset)}")

            if max_samples is not None:
                filtered_dataset = filtered_dataset.select(range(min(max_samples, len(filtered_dataset))))
                
            return filtered_dataset

        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class LiveCodeBenchEvaluator:
    """LiveCodeBench Evaluation Main Class - With Speed Calculation"""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize evaluator"""
        self.config = config
        self.model_interface = LocalModelInterface(config)
        self.dataset_loader = DatasetLoader()

    def run(self) -> tuple[EvaluationSummary, str]:
        """Execute evaluation process"""
        try:
            # Load dataset
            dataset = self.dataset_loader.load_dataset(self.config.max_samples)
            logger.info(f"Dataset loaded successfully, total {len(dataset)} samples")

            # Create output directory
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # Generate code
            logger.info(f"Starting code generation (k={self.config.k})...")
            results = []
            total_time = 0.0
            total_tokens = 0
            for sample in tqdm(dataset, desc="Code Generation Progress"):
                result = self._generate_single_sample(sample)
                results.append(result)
                
                # Accumulate total time and total tokens
                total_time += sum(result.time_taken)
                total_tokens += sum(result.token_counts)
            
            # Save results
            output_file = self._save_results(results, output_path)
            
            # Calculate average speed (token/s)
            average_speed = total_tokens / total_time if total_time > 0 else 0
            
            # Generate summary
            model_name = os.path.basename(self.config.model_path)
            summary = EvaluationSummary(
                model=model_name,
                total_generation_time=total_time,
                total_tokens_generated=total_tokens,
                average_speed=average_speed,
                total_samples=len(results),
                average_tokens_per_sample=total_tokens / len(results) if results else 0
            )
            
            # Record speed information
            speed_info = f"Total Time: {total_time:.2f}s, Total Tokens: {total_tokens}, Average Speed: {average_speed:.2f} token/s"
            append_to_file(f"{model_name}_speed.txt", speed_info)
            
            # Print summary
            self._print_summary(summary, output_file)
            
            # Run evaluator
            if self.config.run_evaluator:
                self.run_evaluator(output_file)
                
            return summary, output_file

        except Exception as e:
            if isinstance(e, LiveCodeBenchEvaluationError):
                raise
            raise LiveCodeBenchEvaluationError(f"Error during evaluation: {e}") from e

    def _generate_single_sample(self, sample: dict[str, Any]) -> GenerationResult:
        """Generate k code snippets for a single sample, record token count and time for each generation"""
        generated_codes = []
        token_counts = []
        time_taken = []
        
        for _ in range(self.config.k):
            code, tokens, time_spent = self.model_interface.generate_code(
                sample, self.config.generation_config
            )
            generated_codes.append(code)
            token_counts.append(tokens)
            time_taken.append(time_spent)

        return GenerationResult(
            question_id=str(sample["question_id"]),
            code_list=generated_codes,
            token_counts=token_counts,
            time_taken=time_taken
        )

    def _save_results(self, results: list[GenerationResult], output_path: Path) -> str:
        """Save generation results, including token count and time information"""
        detailed_output = [
            {
                "question_id": r.question_id,
                "code_list": r.code_list,
                "token_counts": r.token_counts,
                "time_taken": r.time_taken
            }
            for r in results
        ]
        
        model_name = os.path.basename(self.config.model_path)
        params = self.config.generation_config
        output_filename = f"output_{model_name}_t{params.temperature}_k{self.config.k}.json"
        output_file = output_path / output_filename
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, ensure_ascii=False, indent=2)
            
        return str(output_file.absolute())

    def run_evaluator(self, output_file: str) -> None:
        """Run evaluator"""
        logger.info("Starting evaluation of generated code...")
        
        command = [
            "python", "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", output_file
        ]
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd="/xxx/xxx/LiveCodeBench"
            )
            
            logger.info("Evaluator Output:\n" + result.stdout)
            with open(f"{output_file}.eval_results.txt", "w") as f:
                f.write(result.stdout)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluator run failed: {e.stderr}")

    @staticmethod
    def _print_summary(summary: EvaluationSummary, output_file: str) -> None:
        """Print summary information, including token generation speed"""
        logger.info("=" * 70)
        logger.info(f"Evaluation Completed!")
        logger.info(f"Model: {summary.model}")
        logger.info(f"Total Samples: {summary.total_samples}")
        logger.info(f"Total Generation Time: {summary.total_generation_time:.2f} seconds")
        logger.info(f"Total Generated Tokens: {summary.total_tokens_generated}")
        logger.info(f"Average Generation Speed: {summary.average_speed:.2f} token/second")
        logger.info(f"Average Tokens Per Sample: {summary.average_tokens_per_sample:.2f}")
        logger.info(f"Result File: {output_file}")
        logger.info("=" * 70)


# ================================ Command Line Interface ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Single-GPU Version of LiveCodeBench Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--no-evaluator", action="store_true", help="Do not run the evaluator")
    parser.add_argument("--model-path", type=str, help="Model path")
    parser.add_argument("--device", type=str, default="cuda:7", help="Inference device, default: cuda:0")
    parser.add_argument("--max-tokens", type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--top-p", type=float, help="Top-p parameter")
    parser.add_argument("--top-k", type=int, help="Top-k parameter")
    parser.add_argument("--k", type=int, default=1, help="k value in pass@k")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples")

    return parser


def setup_logging(log_level: str) -> None:
    """Configure logging"""
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

    # Generate configuration
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

    # Evaluation configuration
    config = EvaluationConfig(
        run_evaluator=not args.no_evaluator,
        k=args.k,
        generation_config=gen_config,
        max_samples=args.max_samples,
        device=args.device
    )
    
    if args.model_path:
        config.model_path = args.model_path

    try:
        evaluator = LiveCodeBenchEvaluator(config)
        summary, output_file = evaluator.run()
        print(f"\n🎉 Evaluation Completed! Average Generation Speed: {summary.average_speed:.2f} token/second")
        print(f"📝 Results Saved To: {output_file}")

    except LiveCodeBenchEvaluationError as e:
        logger.error(f"Evaluation Failed: {e}")
        exit(1)


# Logging configuration
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()