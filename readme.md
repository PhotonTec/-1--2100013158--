# GPT-2 Fine-tuning Project

This project fine-tunes the GPT-2 model from Hugging Face using the WikiText-2 train dataset and evaluates the perplexity on the WikiText-2 test dataset. The repository includes scripts for training (`train.py`), testing (`test.py`), and a PDF report summarizing the project. 

## Getting Started

### Prerequisites

- Python 3.8
- Pip (Python package installer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Torch 1.9

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PhotonTec/GM-hw1.git
   cd GM-hw1

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

#### Training

Run the training script to fine-tune the GPT-2 model:

```bash
python train.py --data_path path/to/wikitext-2/train --output_dir path/to/save/checkpoints
```

- `--data_path`: Path to the WikiText-2 training dataset.
- `--output_dir`: Directory to save the fine-tuned model checkpoints.

#### Testing

Evaluate the perplexity of the fine-tuned model on the WikiText-2 test dataset:

```bash
python test.py --model_path path/to/saved/checkpoints --test_data_path path/to/wikitext-2/test
```

- `--model_path`: Path to the saved fine-tuned model checkpoints.
- `--test_data_path`: Path to the WikiText-2 test dataset.

### Project Structure

- `train.py`: Script for fine-tuning the GPT-2 model.
- `test.py`: Script for evaluating the perplexity on the test dataset.
- `report.pdf`: PDF report summarizing the project.

## Results

Include any relevant results or findings from your experiments.

## Contributing

Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).

## Acknowledgments

- This is homework1 of generative model class
- Author:2100013158 Xu Tianyi# GM-hw1
