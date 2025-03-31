# Fine-Tuning of LLMs with Hugging Face

This repository contains a Jupyter Notebook for fine-tuning Large Language Models (LLMs) using Hugging Face's `transformers` and related libraries. The notebook guides users through setting up the environment, loading datasets, and fine-tuning models.

## Features
- Step-by-step guide for fine-tuning LLMs.
- Uses Hugging Face `datasets` and `transformers`.
- Integration with `peft` for parameter-efficient tuning.
- Utilizes `bitsandbytes` for low-memory optimization.

## Installation
To set up the required dependencies, run the following command:

```bash
pip uninstall accelerate peft bitsandbytes transformers trl -y
pip install accelerate peft==0.13.2 bitsandbytes transformers trl==0.12.0
pip install huggingface_hub
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook SDS_Fine_Tuning_of_LLMs_Partial_Code.ipynb
   ```
3. Follow the steps in the notebook to fine-tune an LLM.

## Dataset
This fine-tuning process makes use of the [Wiki Medical Terms Dataset](https://huggingface.co/datasets/aboonaji/wiki_medical_terms_llam2_format). This dataset is pre-formatted for LLaMA 2 fine-tuning and contains structured medical terminology extracted from Wikipedia, making it ideal for training language models in the medical domain.

To load the dataset in your notebook, use the following code:

```python
from datasets import load_dataset

dataset = load_dataset("aboonaji/wiki_medical_terms_llam2_format")
print(dataset)
```

## Model
For fine-tuning, we utilize the [LLaMA 2 fine-tuned model](https://huggingface.co/aboonaji/llama2finetune-v2). This model has been further trained on domain-specific datasets, making it more suitable for specialized NLP tasks.

To load the model in your notebook, use the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "aboonaji/llama2finetune-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Requirements
- Python 3.8+
- Jupyter Notebook
- Hugging Face Transformers
- PyTorch or TensorFlow

## Contributing
Feel free to fork the repository and submit pull requests for improvements or additional features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

