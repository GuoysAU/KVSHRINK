"""
Task Data Loading and Formatting

Provides unified interface for loading different evaluation tasks (BoolQ, XSum, etc.)
Simple and practical for research experiments.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TaskExample:
    """
    Single example from a task dataset.
    
    Attributes:
        prompt: Formatted prompt for the model
        label: Ground truth label/answer
        metadata: Additional information (e.g., question, passage, etc.)
    """
    prompt: str
    label: str
    metadata: Dict[str, Any]


# ============================================================================
# Base Task Class
# ============================================================================

class Task:
    """
    Base class for evaluation tasks.
    
    All tasks implement:
    - load_data(): Load dataset from file
    - format_prompt(): Create model input
    - parse_output(): Extract answer from model output
    """
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None, shuffle: bool = False):
        """
        Initialize task.
        
        Args:
            data_path: Path to dataset file
            max_samples: Maximum number of samples to use (None = all)
            shuffle: Whether to shuffle the data
        """
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.examples: List[TaskExample] = []
    
    def load_data(self) -> List[TaskExample]:
        """
        Load dataset from file.
        
        Returns:
            List of TaskExample objects
        """
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def format_prompt(self, example: TaskExample) -> str:
        """
        Format a single example into a prompt for the model.
        
        Args:
            example: TaskExample object
        
        Returns:
            Formatted prompt string
        """
        return example.prompt
    
    def parse_output(self, output: str) -> str:
        """
        Parse model output to extract the answer.

        Args:
            output: Raw model output string

        Returns:
            Parsed answer
        """
        # Default: return cleaned output
        return output.strip()

    def supports_scoring(self) -> bool:
        """是否支持 scoring（似然度打分）评估方式"""
        return False

    def scoring_mode(self) -> str:
        """
        Scoring 模式，支持:
        - "shared_context": 默认，所有 options 共享同一个 context
        - "per_option_context": 每个 option 拥有各自独立的 context (如 WinoGrande)
        """
        return "shared_context"

    def get_scoring_inputs(self, example: TaskExample) -> tuple:
        """
        获取 scoring 评估所需的输入。

        Args:
            example: TaskExample 对象

        Returns:
            (context, options, label_idx) 元组
            - context: 上下文字符串（用于 prefill）
            - options: 选项列表
            - label_idx: 正确选项的索引
        """
        raise NotImplementedError("Subclass must implement get_scoring_inputs() if supports_scoring() returns True")

    def stop_strings(self) -> List[str]:
        """Generation stop strings. Empty list means generate until max_new_tokens."""
        return []

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# BoolQ Task
# ============================================================================

class BoolQTask(Task):
    """
    BoolQ: Boolean Questions (Yes/No reading comprehension)
    
    Format:
        Passage: [passage text]
        
        Question: [question]
        
        Answer (True/False):
    
    Expected output: "True" or "False"
    """
    
    def load_data(self) -> List[TaskExample]:
        """
        Load BoolQ dataset from JSONL file.
        
        Expected format per line:
        {
            "question": "...",
            "passage": "...",
            "answer": true/false
        }
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                
                # Format prompt
                prompt = self._format_boolq_prompt(
                    passage=item['passage'],
                    question=item['question']
                )
                
                # Convert boolean to string
                label = "True" if item['answer'] else "False"
                
                example = TaskExample(
                    prompt=prompt,
                    label=label,
                    metadata={
                        'question': item['question'],
                        'passage': item['passage'],
                    }
                )
                examples.append(example)
        
        # Shuffle if requested
        if self.shuffle:
            import random
            random.shuffle(examples)
        
        # Limit samples
        if self.max_samples is not None:
            examples = examples[:self.max_samples]
        
        self.examples = examples
        return examples
    
    def _format_boolq_prompt(self, passage: str, question: str) -> str:
        """Format BoolQ prompt."""
        return (
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            "Answer (True/False):"
        )
    
    def parse_output(self, output: str) -> str:
        """
        Parse model output to extract True/False answer.
        
        Handles various formats:
        - "True" / "False"
        - "true" / "false"
        - "Yes" / "No"
        - Output with explanation followed by answer
        """
        output = output.strip().lower()
        
        # Check for explicit true/false
        if 'true' in output[:20]:  # Check first 20 chars
            return "True"
        elif 'false' in output[:20]:
            return "False"
        
        # Check for yes/no
        if 'yes' in output[:20]:
            return "True"
        elif 'no' in output[:20]:
            return "False"
        
        # Default: return first word
        first_word = output.split()[0] if output.split() else ""
        if first_word in ['true', 'yes']:
            return "True"
        elif first_word in ['false', 'no']:
            return "False"
        
        # Unable to parse, return raw
        return output.capitalize()

    def supports_scoring(self) -> bool:
        return True

    def get_scoring_inputs(self, example: TaskExample) -> tuple:
        context = (
            f"{example.metadata['passage']}\n"
            f"Question: {example.metadata['question']}?\n"
            "Answer:"
        )
        label_idx = 1 if example.label == "True" else 0
        return context, [" no", " yes"], label_idx



# ============================================================================
# XSum Task
# ============================================================================

class XSumTask(Task):
    """
    XSum: Extreme Summarization (generate one-sentence summary)
    
    Format:
        Article: [article text]
        
        Summarize the above article in one sentence:
    
    Expected output: One sentence summary
    """
    
    def load_data(self) -> List[TaskExample]:
        """
        Load XSum dataset from JSONL file.
        
        Expected format per line:
        {
            "document": "...",
            "summary": "..."
        }
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                
                # Format prompt
                prompt = self._format_xsum_prompt(document=item['document'])
                
                example = TaskExample(
                    prompt=prompt,
                    label=item['summary'],
                    metadata={
                        'document': item['document'],
                    }
                )
                examples.append(example)
        
        # Shuffle if requested
        if self.shuffle:
            import random
            random.shuffle(examples)
        
        # Limit samples
        if self.max_samples is not None:
            examples = examples[:self.max_samples]
        
        self.examples = examples
        return examples
    
    def _format_xsum_prompt(self, document: str) -> str:
        """Format XSum prompt."""
        return (
            f"Article: {document}\n\n"
            "Summarize the above article in one sentence:"
        )
    
    def parse_output(self, output: str) -> str:
        """
        Parse model output to extract summary.
        
        Returns the first sentence or first line.
        """
        output = output.strip()
        
        # Split by newline and take first non-empty line
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if lines:
            return lines[0]
        
        return output


class XSumUnitxtTask(Task):
    """
    XSum with Unitxt-compatible prompt format, loaded from local dev.jsonl.
    Prompt matches: card=cards.xsum,template=templates.summarization.abstractive.full
    """

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                document = item["document"]
                summary = item["summary"]
                # Unitxt template: "Summarize the following document: {text}\n"
                prompt = f"Summarize the following document: {document}\n"
                examples.append(
                    TaskExample(
                        prompt=prompt,
                        label=summary,
                        metadata={"document": document},
                    )
                )

        if self.shuffle:
            import random
            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[:self.max_samples]

        self.examples = examples
        return examples

    def parse_output(self, output: str) -> str:
        output = output.strip()
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if lines:
            return lines[0]
        return output


class XSumLMEvalTask(Task):
    """
    XSum aligned with Palu's lm_eval standard.

    Differences from XSumUnitxtTask:
    - Prompt ends with '.' (matches unitxt templates.summarization.abstractive.full)
    - Metric uses use_stemmer=False (matches unitxt metrics.rouge default)
    - parse_output takes first line (equivalent to lm_eval's until=["\n"])
    """

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                document = item["document"]
                summary = item["summary"]
                prompt = f"Summarize the following document: {document}."
                examples.append(
                    TaskExample(
                        prompt=prompt,
                        label=summary,
                        metadata={"document": document},
                    )
                )

        if self.shuffle:
            import random
            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[:self.max_samples]

        self.examples = examples
        return examples

    def parse_output(self, output: str) -> str:
        output = output.strip()
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        if lines:
            return lines[0]
        return output


# ============================================================================
# GSM8K Task
# ============================================================================

class GSM8KTask(Task):
    """
    Grade School Math word problems with 0-shot prompting.

    使用 0-shot 评估以便公平比较 baseline 和 compressed 的性能差异。
    其他论文（如 PyramidKV）也使用 0-shot 进行评估。
    """

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question", "").strip()
                answer_raw = item.get("answer", "")
                label = self._extract_final_answer(answer_raw)

                # Create 0-shot prompt
                prompt = self._create_cot_prompt(question)

                examples.append(
                    TaskExample(
                        prompt=prompt,
                        label=label,
                        metadata={"question": question, "answer_raw": answer_raw},
                    )
                )

        if self.shuffle:
            import random
            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[: self.max_samples]

        self.examples = examples
        return examples

    def _create_cot_prompt(self, question: str) -> str:
        """
        Create a 0-shot prompt for GSM8K.

        强制模型只输出最终数字，避免冗长推理和提取失败。
        """
        return f"Q: {question}\nA: (Only output the final number)"

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """
        Extract the final numeric answer from GSM8K response.

        Handles various formats:
        - "#### 2,125" (original GSM8K format with comma)
        - "The answer is 42."
        - "Final answer: 42"
        - "= 42"
        - "42" (just the number)

        Supports numbers with thousand separators (e.g., 2,125 -> 2125).
        """
        def clean_number(num_str: str) -> str:
            """Remove thousand separators and clean up number string."""
            return num_str.replace(",", "").strip()

        # Pattern for numbers with optional thousand separators and decimals
        # Matches: 42, 2,125, 114,200, 3.14, -42, -2,125
        num_pattern = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"

        # Priority 1: "#### <number>" format (GSM8K standard)
        if "####" in text:
            after_marker = text.split("####")[-1].strip()
            match = re.search(num_pattern, after_marker)
            if match:
                return clean_number(match.group())

        # Priority 2: "The answer is <number>" pattern
        answer_pattern = re.search(
            rf"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*({num_pattern})",
            text, re.IGNORECASE
        )
        if answer_pattern:
            return clean_number(answer_pattern.group(1))

        # Priority 3: "Final answer: <number>" pattern
        final_pattern = re.search(
            rf"final\s+answer[:\s]*({num_pattern})",
            text, re.IGNORECASE
        )
        if final_pattern:
            return clean_number(final_pattern.group(1))

        # Priority 4: "= <number>" at end of expression
        equals_pattern = re.search(
            rf"=\s*({num_pattern})\s*$",
            text.strip(), re.MULTILINE
        )
        if equals_pattern:
            return clean_number(equals_pattern.group(1))

        # Fallback: Find all numbers and return the last one
        matches = re.findall(num_pattern, text)
        if matches:
            return clean_number(matches[-1])

        return text.strip()

    def parse_output(self, output: str) -> str:
        """Parse model output to extract the final numeric answer."""
        return self._extract_final_answer(output)

# ============================================================================
# WinoGrande Task
# ============================================================================

class WinoGrandeTask(Task):
    """
    WinoGrande: Commonsense reasoning via pronoun resolution.

    Fill in the blank with the correct option (1 or 2).
    """

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                sentence = item["sentence"]
                option1 = item["option1"]
                option2 = item["option2"]
                answer = item["answer"]  # "1" or "2"

                prompt = self._format_prompt(sentence, option1, option2)

                examples.append(
                    TaskExample(
                        prompt=prompt,
                        label=answer,  # "1" or "2"
                        metadata={
                            "sentence": sentence,
                            "option1": option1,
                            "option2": option2,
                        }
                    )
                )

        if self.shuffle:
            import random
            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[: self.max_samples]

        self.examples = examples
        return examples

    def _format_prompt(self, sentence: str, option1: str, option2: str) -> str:
        """Format WinoGrande prompt."""
        return (
            "Fill in the blank (_) with the correct option.\n\n"
            f"Sentence: {sentence}\n"
            f"1: {option1}\n"
            f"2: {option2}\n\n"
            "Answer with only 1 or 2:"
        )

    def parse_output(self, output: str) -> str:
        """Extract 1 or 2 from model output."""
        output = output.strip()
        if "1" in output and "2" not in output:
            return "1"
        if "2" in output and "1" not in output:
            return "2"
        # 取第一个出现的数字
        for char in output:
            if char in ("1", "2"):
                return char
        return output

    def supports_scoring(self) -> bool:
        return True

    def scoring_mode(self) -> str:
        return "per_option_context"

    def get_scoring_inputs(self, example: TaskExample) -> tuple:
        """
        WinoGrande scoring: P(suffix | prefix + option)

        Returns: (contexts, continuations, label_idx)
        """
        sentence = example.metadata["sentence"]
        option1 = example.metadata["option1"]
        option2 = example.metadata["option2"]
        label_idx = int(example.label) - 1  # "1"/"2" -> 0/1

        # 分割句子为前缀和后缀
        if "_" in sentence:
            parts = sentence.split("_", 1)
            sentence_prefix = parts[0]
            sentence_suffix = parts[1] if len(parts) > 1 else ""
        else:
            sentence_prefix = sentence
            sentence_suffix = ""

        contexts = [
            sentence_prefix + option1,
            sentence_prefix + option2,
        ]
        
        continuations = [
            sentence_suffix,
            sentence_suffix,
        ]
        
        return contexts, continuations, label_idx


# ============================================================================
# HellaSwag Task
# ============================================================================

class HellaSwagTask(Task):
    """
    HellaSwag: Commonsense reasoning via sentence completion.

    Choose the most plausible continuation from 4 options.
    """

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                ctx = item["ctx"]  # Context
                endings = item["endings"]  # List of 4 possible endings
                label_idx = int(item["label"])  # 0-3

                # Format prompt
                prompt = self._format_hellaswag_prompt(ctx, endings)

                # Label is the 1-indexed number as string (to match prompt format)
                label = str(label_idx + 1)  # Convert 0-3 to "1"-"4"

                examples.append(
                    TaskExample(
                        prompt=prompt,
                        label=label,
                        metadata={
                            "context": ctx,
                            "endings": endings,
                            "label_idx": label_idx,
                            "ctx_a": item["ctx_a"],
                            "ctx_b": item["ctx_b"],
                            "activity_label": item["activity_label"],
                        }
                    )
                )

        if self.shuffle:
            import random
            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[: self.max_samples]

        self.examples = examples
        return examples

    def _format_hellaswag_prompt(self, context: str, endings: List[str]) -> str:
        """Format HellaSwag prompt."""
        endings_text = "\n".join([f"{i+1}. {ending}" for i, ending in enumerate(endings)])
        return (
            "Choose the most plausible continuation for the following context.\n\n"
            f"Context: {context}\n\n"
            "Options:\n"
            f"{endings_text}\n\n"
            "Which option is the most plausible continuation? Answer with only the number (1-4).\n\n"
            "Answer:"
        )

    def parse_output(self, output: str) -> str:   ### option 1: Generation 模式
        """Parse model output - extract number 1-4."""
        output = output.strip()

        # Try to extract number 1-4
        import re
        match = re.search(r'\b([1-4])\b', output)
        if match:
            return match.group(1)  # Return "1"-"4" to match label format

        return output

    def supports_scoring(self) -> bool:
        return True


    def get_scoring_inputs(self, example: TaskExample) -> tuple:
        ctx = example.metadata["context"]
        endings = example.metadata["endings"]
        label_idx = example.metadata["label_idx"]

        return ctx, [" " + e for e in endings], label_idx




# ============================================================================
# OpenBookQA Task
# ============================================================================

class OpenBookQATask(Task):
    """Multiple-choice science questions from OpenBookQA."""

    def load_data(self) -> List[TaskExample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question")
                choices = item.get("choices", [])
                answer = item.get("answer")

                prompt = self._format_openbook_prompt(question, choices)
                example = TaskExample(
                    prompt=prompt,
                    label=answer,
                    metadata={
                        "question": question,
                        "choices": choices,
                        "id": item.get("id"),
                    },
                )
                examples.append(example)

        if self.shuffle:
            import random

            random.shuffle(examples)

        if self.max_samples is not None:
            examples = examples[: self.max_samples]

        self.examples = examples
        return examples

    def _format_openbook_prompt(self, question: str, choices: List[Dict[str, str]]) -> str:
        lines = ["Question: " + question, "Choices:"]
        for choice in choices:
            lines.append(f"  ({choice['label']}) {choice['text']}")
        lines.append("Answer (A/B/C/D):")
        return "\n".join(lines)

    def parse_output(self, output: str) -> str:
        output = output.strip().upper()
        for option in ["A", "B", "C", "D"]:
            if output.startswith(option):
                return option
        # fallback: first non-empty character
        for ch in output:
            if ch in "ABCD":
                return ch
        return output[:1]

    def supports_scoring(self) -> bool:
        return True

    def get_scoring_inputs(self, example: TaskExample) -> tuple:
        question = example.metadata["question"]
        choices = example.metadata["choices"]

        # context 包含完整的问题和选项列表
        choices_text = "\n".join([f"  ({c['label']}) {c['text']}" for c in choices])
        context = f"Question: {question}\nChoices:\n{choices_text}\nAnswer:"
        # options 只打分短选项标签
        options = ["A", "B", "C", "D"]
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        label_idx = label_map.get(example.label, 0)


        # context = question
        # options = [f" {c['text']}" for c in choices]

        # label_map = {c["label"]: i for i, c in enumerate(choices)}
        # label_idx = label_map[example.label]


        return context, options, label_idx


# ============================================================================
# Task Factory
# ============================================================================

DEFAULT_DATA_PATHS = {
    'boolq': 'data/boolq/dev.jsonl',
    'gsm8k': 'data/gsm8k/dev.jsonl',
    'xsum': 'data/xsum/dev.jsonl',
    'xsum_unitxt': 'data/xsum/dev.jsonl',
    'xsum_lmeval': 'data/xsum/test.jsonl',
    'openbookqa': 'data/openbookqa/dev.jsonl',
    'winogrande': 'data/winogrande/dev.jsonl',
    'hellaswag': 'data/hellaswag/dev.jsonl',
}


TASK_REGISTRY = {
    'boolq': BoolQTask,
    'xsum': XSumTask,
    'xsum_unitxt': XSumUnitxtTask,
    'xsum_lmeval': XSumLMEvalTask,
    'openbookqa': OpenBookQATask,
    'gsm8k': GSM8KTask,
    'winogrande': WinoGrandeTask,
    'hellaswag': HellaSwagTask,
}


def create_task(task_name: str, data_path: Optional[str] = None, max_samples: Optional[int] = None, 
                shuffle: bool = False) -> Task:
    """
    Create task instance by name.
    
    Args:
        task_name: Name of the task ("boolq", "xsum", etc.)
        data_path: Path to data file
        max_samples: Maximum samples to load
        shuffle: Whether to shuffle data
    
    Returns:
        Task instance
    
    Raises:
        ValueError: If task name is not recognized
    
    Example:
        task = create_task("boolq", "data/boolq/dev.jsonl", max_samples=100)
        task.load_data()
        for example in task:
            prompt = example.prompt
            label = example.label
    """
    task_name = task_name.lower()
    
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    
    if not data_path:
        data_path = DEFAULT_DATA_PATHS.get(task_name)
    if not data_path:
        raise ValueError(f"No data path provided for task '{task_name}' and no default known.")

    task_class = TASK_REGISTRY[task_name]
    task = task_class(data_path=data_path, max_samples=max_samples, shuffle=shuffle)
    # Store task name for downstream metric selection (e.g., ROUGE vs accuracy)
    task.task_name = task_name
    
    return task


def create_task_from_config(config) -> Task:
    """
    Create task from Config object.
    
    Args:
        config: Config object with task_name, data_path, max_samples, shuffle
    
    Returns:
        Task instance with data loaded
    
    Example:
        from configs import Config
        config = Config(task_name="boolq", data_path="data/boolq/dev.jsonl")
        task = create_task_from_config(config)
    """
    task = create_task(
        task_name=config.task_name,
        data_path=config.data_path,
        max_samples=config.max_samples,
        shuffle=config.shuffle,
    )
    task.load_data()
    return task


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
   pass
