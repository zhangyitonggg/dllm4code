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

# 导入LLaDA特有的生成函数
from llada_generate import generate


# ================================ 配置和常量 ================================

@dataclass(frozen=True)
class GenerationConfig:
    """代码生成配置类，所有生成参数在此控制"""
    temperature: float = 0.1        # 生成温度
    steps: int = 512                # 生成步骤数
    gen_length: int = 512           # 生成长度
    block_length: int = 16         # 块长度
    cfg_scale: float = 0.0          # 分类器引导尺度
    remasking: str = 'low_confidence'  # 重掩码策略


@dataclass
class EvaluationConfig:
    """评估配置类，所有评估参数在此控制"""
    model_path: str = "/xxx/xxx/LLaDA-8B-Instruct"  # LLaDA模型路径
    k: int = 1                     # pass@k中的k值
    output_dir: str = "LiveCodeBench_llada_results"  # 输出目录
    max_samples: int | None = None  # 最大样本数，None表示全部
    max_new_tokens: int = 512      # 最大新生成token数
    timeout_seconds: int = 5       # 超时时间（秒）
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)  # 生成配置
    device: str = "cuda:7"         # 运行设备
    custom_evaluator_path: str = "/xxx/xxx/LiveCodeBench/lcb_runner"  # 评估器路径
    run_evaluator: bool = True     # 是否自动运行评估器


class GenerationResult(NamedTuple):
    """单个生成样本的结果"""
    question_id: str
    code_list: list[str]  # 存储k次生成的代码


class EvaluationSummary(NamedTuple):
    """评估总结结果"""
    model: str
    model_type: str
    generation_time: float
    average_time_per_sample: float
    total_samples: int


# ================================ 自定义异常 ================================

class LiveCodeBenchEvaluationError(Exception):
    """LiveCodeBench评估过程中的基础异常类"""
    pass


class ModelLoadError(LiveCodeBenchEvaluationError):
    """模型加载失败异常"""
    pass


class DatasetLoadError(LiveCodeBenchEvaluationError):
    """数据集加载失败异常"""
    pass


class CodeGenerationError(LiveCodeBenchEvaluationError):
    """代码生成失败异常"""
    pass


# ================================ 辅助函数 ================================

@contextmanager
def multi_timer():
    """多计时器上下文管理器，用于统计总执行时间"""
    total_time = 0.0
    start_time = None
    
    class Timer:
        def start(self):
            nonlocal start_time
            if start_time is not None:
                raise RuntimeError("计时器已经开始，请先停止当前计时")
            start_time = time.perf_counter()
        
        def stop(self):
            nonlocal total_time, start_time
            if start_time is None:
                raise RuntimeError("请先开始计时")
            end_time = time.perf_counter()
            total_time += end_time - start_time
            start_time = None
            
        def get_total(self):
            """获取当前累计的总时间"""
            return total_time
    
    try:
        yield Timer()
    finally:
        print(f"所有计时语句的总执行时间: {total_time:.6f} 秒")


def append_number_to_file(filename, number):
    """将数字追加到文件末尾"""
    try:
        with open(filename, 'a') as file:
            file.write(f"{number}\n")
        print(f"成功将数字 {number} 追加到文件 {filename}")
    except Exception as e:
        print(f"追加数字时出错: {e}")


# ================================ 核心功能类 ================================

class ModelInterface:
    """模型接口类，封装LLaDA模型加载和代码生成功能"""

    def __init__(self, model_path: str, device: str) -> None:
        """初始化模型接口"""
        self.model_path = model_path
        self.device = device
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        # LLaDA特有的EOS标记
        self._eos_markers = [
            "<|endoftext|>",
            "<|eot_id|>",
            "\n```",
            "\nassert ",
            "# Example usage:",
        ]
        self._load_model()

    def _load_model(self) -> None:
        """加载LLaDA模型和分词器"""
        logger.info(f"正在加载LLaDA模型: {self.model_path}")

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
            
            # 移动到指定设备
            self._model = self._model.to(self.device).eval()
            logger.info(f"✅ LLaDA模型加载成功，使用设备: {self.device}")

        except Exception as e:
            error_msg = f"LLaDA模型加载失败: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    @property
    def model(self) -> AutoModel:
        """获取模型实例"""
        if self._model is None:
            raise ModelLoadError("模型未正确加载")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """获取分词器实例"""
        if self._tokenizer is None:
            raise ModelLoadError("分词器未正确加载")
        return self._tokenizer

    def generate_code(
        self,
        question:dict,
        config: GenerationConfig,
        max_new_tokens: int = 512,
    ) -> str:
        """生成代码（LLaDA特有方法）"""
        # 构建提示模板 - 适配LLaDA的格式
        def get_question_template_answer(question):
            if question["starter_code"]:
                inner = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.```python  ```. You might only need to fill in the given function and return the given list of number. Here is the starter code:" + "\n" + question["starter_code"]
            else:
                inner = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n"
            
            prompt_template = f"""<|startoftext|><|start_header_id|>user<|end_header_id|>

You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question["question_content"]}
{inner}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```python
"""
            if question["starter_code"]:
                prompt_template += question["starter_code"]
            
            return prompt_template
        
        prompt_template = get_question_template_answer(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_template,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # 生成代码 - 使用LLaDA特有的generate函数
        with torch.no_grad():
            output = generate(
                self.model,
                input_ids,
                steps=config.steps,
                gen_length=config.gen_length,
                block_length=config.block_length,
                temperature=config.temperature,
                cfg_scale=config.cfg_scale,
                remasking=config.remasking
            )
        
        # 解码生成的内容（排除输入部分）
        generated_text = self.tokenizer.batch_decode(
            output[:, input_ids.shape[1]:],  # 只取新生成的部分
            skip_special_tokens=True
        )[0]
        print(generated_text)
        # 找到最早出现的EOS标记并截取
        eos_markers = copy.deepcopy(self._eos_markers)
        if question["starter_code"]:
            eos_markers.extend(["\nif __name__","\ndef main(","\nprint("])
        min_index = None
        for marker in eos_markers:
            index = generated_text.find(marker)
            if index != -1:
                if min_index is None or index < min_index:
                    min_index = index
        
        # 截取结果
        if min_index is not None:
            extracted_code = generated_text[:min_index]
        else:
            extracted_code = generated_text  # 如果没有找到EOS标记，使用全部内容
            
        # 组合 starter code 和生成的代码
        if question["starter_code"]:
            full_code = question["starter_code"] + "\n" + extracted_code
        else:
            full_code = extracted_code
        print(full_code)
        return full_code


class DatasetLoader:
    """数据集加载器类"""

    @staticmethod
    @lru_cache(maxsize=1)
    def load_LiveCodeBench_dataset(max_samples: int | None = None) -> datasets.Dataset:
        """加载LiveCodeBench数据集（带缓存），并筛选2024年10月之后的样本"""
        try:
            # 加载原始数据集
            dataset = load_dataset("/xxx/xxx/code_generation_lite", version_tag="release_v6", trust_remote_code=True)["test"]
            logger.info(f"✅ 原始LiveCodeBench数据集加载成功，共{len(dataset)}个样本")

            # 定义日期筛选函数：只保留2024年10月1日及之后的样本
            def filter_by_date(example):
                # 获取样本中的比赛日期
                contest_date_str = example.get("contest_date")
                if not contest_date_str:  # 无日期信息的样本排除
                    return False
                
                try:
                    # 解析日期（格式如：'2023-08-21T00:00:00'）
                    contest_date = datetime.strptime(contest_date_str, '%Y-%m-%dT%H:%M:%S')
                    # 设定筛选阈值：2024年10月1日
                    cutoff_date = datetime(1970, 1, 1)
                    # 保留日期在阈值之后的样本
                    return contest_date >= cutoff_date
                except ValueError:  # 日期格式错误的样本排除
                    logger.warning(f"日期格式错误: {contest_date_str}，已排除该样本")
                    return False

            # 应用筛选
            filtered_dataset = dataset.filter(filter_by_date)
            logger.info(f"筛选后（2024年10月之后）的样本数: {len(filtered_dataset)}")

            # 处理最大样本数限制
            if max_samples is not None:
                actual_max = min(max_samples, len(filtered_dataset))
                filtered_dataset = filtered_dataset.select(range(actual_max))
                logger.info(f"已限制最大样本数为: {actual_max}")

            return filtered_dataset

        except Exception as e:
            error_msg = f"LiveCodeBench数据集加载失败: {e}"
            logger.error(error_msg)
            raise DatasetLoadError(error_msg) from e


class LiveCodeBenchGenerator:
    """LiveCodeBench代码生成器主类"""

    def __init__(self, config: EvaluationConfig) -> None:
        """初始化生成器"""
        self.config = config
        self.model_interface = ModelInterface(config.model_path, config.device)
        self.dataset_loader = DatasetLoader()

    def generate(self) -> tuple[EvaluationSummary, str]:
        """执行代码生成并保存结果"""
        try:
            # 加载数据集
            dataset = self.dataset_loader.load_LiveCodeBench_dataset(self.config.max_samples)
            logger.info(f"成功加载数据集，共{len(dataset)}个样本")

            # 创建输出目录
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)

            # 开始生成代码
            logger.info(f"开始使用LLaDA模型生成代码（k={self.config.k}）...")
            results: list[GenerationResult] = []
            start_time = time.time()
            
            # 初始化计时器
            with multi_timer() as timer:
                for i, sample in enumerate(tqdm(dataset, desc="代码生成进度")):
                    # 计时每个样本的处理
                    timer.start()
                    result = self._generate_single_sample(sample)
                    timer.stop()
                    
                    results.append(result)

                    # 定期打印进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"进度: {i+1}/{len(dataset)}")
            
            # 保存总时间
            total_time = timer.get_total()
            
            # 保存生成结果并返回统计信息
            output_file = self._save_generation_results(results, start_time, output_path)
            
            # 记录生成速度
            speed_info = f"{total_time} {output_file}"
            append_number_to_file("llada_speed.txt", speed_info)
            
            # 生成总结信息
            summary = EvaluationSummary(
                model=self.config.model_path,
                model_type="llada",
                generation_time=total_time,
                average_time_per_sample=total_time / len(results) if results else 0,
                total_samples=len(results)
            )
            
            # 打印结果摘要
            self._print_generation_summary(summary, output_file)
            
            # 如果需要，运行自定义评估器
            if self.config.run_evaluator:
                self.run_custom_evaluator(output_file)
                
            return summary, output_file

        except Exception as e:
            if isinstance(e, LiveCodeBenchEvaluationError):
                raise
            error_msg = f"生成过程中发生未预期错误: {e}"
            logger.error(error_msg)
            raise LiveCodeBenchEvaluationError(error_msg) from e

    def _generate_single_sample(self, sample: dict[str, Any]) -> GenerationResult:
        """为单个样本生成k次代码"""
        # 进行k次代码生成
        generated_codes = []
        for i in range(self.config.k):
            # 生成代码
            try:
                generated_content = self.model_interface.generate_code(
                    sample, self.config.generation_config, self.config.max_new_tokens
                )

                # 代码清理
                generated_code = generated_content
                logger.debug(f"生成的代码: {generated_code}")
            except Exception as e:
                logger.warning(f"任务{sample['question_id']}第{i+1}次代码生成失败: {e}")
                generated_code = "# 代码生成失败\npass"

            generated_codes.append(generated_code)

        return GenerationResult(
            question_id=str(sample["question_id"]),
            code_list=generated_codes
        )

    def _save_generation_results(
        self,
        results: list[GenerationResult],
        start_time: float,
        output_path: Path,
    ) -> str:
        """保存生成结果为符合自定义评估器要求的格式"""
        # 转换为要求的JSON格式
        output_data = [
            {
                "question_id": result.question_id,
                "code_list": result.code_list
            }
            for result in results
        ]
        
        # 生成文件名（包含生成参数）
        params = self.config.generation_config
        output_filename = f"output_llada_steps{params.steps}_gen{params.gen_length}_block{params.block_length}_temp{params.temperature}_k{self.config.k}.json"
        output_file = output_path / output_filename
        
        # 保存文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return str(output_file.absolute())

    def run_custom_evaluator(self, output_file: str) -> None:
        """运行LiveCodeBench的自定义评估器"""
        logger.info(f"开始使用自定义评估器评估生成的代码...")
        
        # 构建评估命令
        evaluator_script = os.path.join(self.config.custom_evaluator_path, "runner", "custom_evaluator.py")
        command = [
            "python", "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", output_file
        ]
        
        try:
            # 运行评估器
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd="/xxx/xxx/LiveCodeBench"
            )
            
            # 输出评估结果
            logger.info("评估器输出:")
            logger.info(result.stdout)
            
            # 保存评估结果
            eval_result_file = f"{output_file}.eval_results.txt"
            with open(eval_result_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)
                
            logger.info(f"评估结果已保存到: {eval_result_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"评估器运行失败: {e.stderr}")
        except Exception as e:
            logger.error(f"运行评估器时发生错误: {e}")

    @staticmethod
    def _print_generation_summary(
        summary: EvaluationSummary,
        output_file: str,
    ) -> None:
        """打印生成结果摘要"""
        logger.info("=" * 50)
        logger.info(f"LLaDA模型代码生成完成！")
        logger.info("=" * 50)
        logger.info(f"模型: {summary.model}")
        logger.info(f"模型类型: LLaDA")
        logger.info(f"总样本数: {summary.total_samples}")
        logger.info(f"生成用时: {summary.generation_time:.2f}秒")
        logger.info(f"平均每样本: {summary.average_time_per_sample:.2f}秒")
        logger.info(f"生成结果文件: {output_file}")
        logger.info("=" * 50)


# ================================ 命令行接口 ================================

def create_arg_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="LiveCodeBench LLaDA模型代码生成器，参数由配置类控制",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别",
    )
    
    parser.add_argument(
        "--no-evaluator",
        action="store_true",
        help="不自动运行自定义评估器",
    )

    return parser


def setup_logging(log_level: str) -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """主函数"""
    parser = create_arg_parser()
    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 创建配置实例
    config = EvaluationConfig(
        run_evaluator=not args.no_evaluator  # 仅这个参数来自命令行
    )

    try:
        # 创建生成器并运行
        generator = LiveCodeBenchGenerator(config)
        summary, output_file = generator.generate()

        # 打印成功信息
        print(f"\n🎉 LLaDA模型代码生成成功完成！")
        print(f"📊 生成样本数: {summary.total_samples}")
        print(f"📝 结果保存在: {output_file}")
        if config.run_evaluator:
            print(f"✅ 已自动运行评估器")

    except LiveCodeBenchEvaluationError as e:
        logger.error(f"生成失败: {e}")
        print(f"\n❌ 代码生成失败: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"未预期的错误: {e}")
        print(f"\n❌ 发生未预期的错误: {e}")
        exit(1)


# ================================ 日志配置 ================================

logger = logging.getLogger(__name__)


# ================================ 主程序入口 ================================

if __name__ == "__main__":
    main()