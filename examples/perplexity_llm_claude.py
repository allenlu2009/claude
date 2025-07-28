import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
#import matplotlib.pyplot as plt
from collections import namedtuple
from dataclasses import dataclass


# Define model-specific differences with max sequence lengths
model_args = {
    "gpt2": {"name": "gpt2", "max_length": 1024},
    "gpt2-large": {"name": "gpt2-large", "max_length": 1024},
    "gpt2-xl": {"name": "gpt2-xl", "max_length": 1024},
    "Qwen2-VL-7B": {"name": "Qwen/Qwen2-VL-7B-Instruct", "max_length": 4096},
    "Qwen2.5-3B": {"name": "Qwen/Qwen2.5-3B-Instruct", "max_length": 32768},
    "gemma-7B": {"name": "google/gemma-7b-it", "max_length": 8192},
    "gemma3-1B": {"name": "google/gemma-3-1b-it", "max_length": 131072},
    "gemma2-2B": {"name": "google/gemma-2-2b-it", "max_length": 131072},
    "gemma3-4B": {"name": "google/gemma-3-4b-it", "max_length": 131072},
    "gemma2-7B": {"name": "google/gemma-2-7b-it", "max_length": 131072},
    "Llama3.2-1B": {"name": "meta-llama/Llama-3.2-1B-Instruct", "max_length": 8192},
    "Llama3.2-3B": {"name": "meta-llama/Llama-3.2-3B", "max_length": 8192},
    "Phi3-mini-4k": {"name": "microsoft/Phi-3-mini-4k-instruct", "max_length": 4096},
    "Phi4-mini": {"name": "microsoft/Phi-4-mini-instruct", "max_length": 131072},
    "Phi4-mini-flash": {"name": "microsoft/Phi-4-mini-flash-reasoning", "max_length": 262144}
}

dataset_args = {
    "Wikitext2": ("wikitext", "wikitext-2-raw-v1"), # Tuple: (dataset_name, config_name)
    "Wikitext103": ("wikitext", "wikitext-103-raw-v1"),
    "Shakespeare": "karpathy/tiny_shakespeare", # Direct dataset identifier
    "C4": ("c4", "en"),  # Adding C4 dataset with English configuration
    "PTB": ("ptb_text_only", "penn_treebank")  # Adding PTB dataset
}

# Load the model and tokenizer
model_arr = ['gpt2-large', 'Llama3.2-1B', 'Llama3.2-3B', 'Phi3-mini-4k', 'Qwen2.5-3B', 'gemma-7B']
# model_arr = ['gemma3-1B', 'gemma3-4B', 'gemma-7B']
model_arr = ['Phi3-mini-4k', 'Phi4-mini', 'Phi4-mini-flash']
#model_arr = ['Phi4-mini-flash']
# model_arr = ['gemma3-4B', 'Phi4-mini', 'gemma-7B']
#model_arr = ['Phi3-mini-4k', 'gemma-7B', 'Qwen-VL-7B']
#model_arr = ['Llama3.2-1B']
#model_arr = ['gpt2', 'gpt2-large', 'gpt2-xl']
# model_arr = ['gpt2-xl']
#model_arr = ['gemma-7B', 'Qwen-VL-7B']
#model_arr = ['Qwen2.5-3B']
# model_arr = ['Llama3.2-3B']
#model_arr = ['Phi4-mini-flash']


# define the dataset
#dataset_name = "Wikitext2"
#dataset_arr = ["PTB"]
dataset_arr = ['Wikitext2', 'PTB', 'Shakespeare']
#dataset_arr = ["C4"]  # too big to download

@dataclass
class ChunkParam:
    block_size: int
    stride_ratio: float  # Changed from stride to stride_ratio
    batch_size: int

# chunk_params = [
#     ChunkParam(block_size=1024, stride_ratio=0.5, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.5, batch_size=1),
#     ChunkParam(block_size=4096, stride_ratio=0.5, batch_size=1)
# ]

# chunk_params = [
#     ChunkParam(block_size=2048, stride_ratio=0.1, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.2, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.3, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.4, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.5, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.6, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.7, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.8, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=0.9, batch_size=1),
#     ChunkParam(block_size=2048, stride_ratio=1.0, batch_size=1)
# ]

chunk_params = [
    # ChunkParam(block_size=2048, stride_ratio=0.25, batch_size=1),
    # ChunkParam(block_size=2048, stride_ratio=0.5,  batch_size=1),
    # ChunkParam(block_size=2048, stride_ratio=0.75, batch_size=1),
    # ChunkParam(block_size=2048, stride_ratio=1.0,  batch_size=1),
    ChunkParam(block_size=4096, stride_ratio=0.25, batch_size=1),
    ChunkParam(block_size=4096, stride_ratio=0.5,  batch_size=1),
    ChunkParam(block_size=4096, stride_ratio=0.75, batch_size=1),
    ChunkParam(block_size=4096, stride_ratio=1.0,  batch_size=1),
    # ChunkParam(block_size=6144, stride_ratio=0.25, batch_size=1),
    # ChunkParam(block_size=6144, stride_ratio=0.5,  batch_size=1),
    # ChunkParam(block_size=6144, stride_ratio=0.75, batch_size=1),
    # ChunkParam(block_size=6144, stride_ratio=1.0,  batch_size=1),
    # ChunkParam(block_size=8192, stride_ratio=0.25, batch_size=1),
    # ChunkParam(block_size=8192, stride_ratio=0.5,  batch_size=1),
    # ChunkParam(block_size=8192, stride_ratio=0.75, batch_size=1),
    # ChunkParam(block_size=8192, stride_ratio=1.0,  batch_size=1),
  ]

max_samples = None  # 如果需要限制處理的樣本數，設置此參數


# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset based on dataset_name
# 將數據集的所有文本拼接為一個長字符串
def dataset2text(dataset_name):
    dataset_arg = dataset_args[dataset_name]

    if dataset_name == "PTB":
        dataset = load_dataset('ptb_text_only', 'penn_treebank')
        dataset = dataset['test']
        #text = " ".join(dataset["sentence"])
        text = " \n ".join(dataset["sentence"])
    elif isinstance(dataset_arg, tuple):  # For Wikitext and Wikitext103
        dataset = load_dataset(dataset_arg[0], dataset_arg[1], split="test")
        #text = " ".join(dataset["text"])
        text = "\n\n".join(dataset["text"])
    else:
        dataset = load_dataset(dataset_arg, split="train")
        #text = " ".join(dataset["text"])
        text = "\n\n".join(dataset["text"])

    return text


# import urllib.request
# import os
# import requests

# def dataset2text(dataset_name):
#     dataset_arg = dataset_args[dataset_name]
#     if dataset_name == "PTB":
#         # Download PTB test data from tomsercu/lstm repository
#         url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
        
#         # Create data directory if it doesn't exist
#         os.makedirs("data", exist_ok=True)
        
#         # Download if not already present
#         if not os.path.exists("data/ptb_test.txt"):
#             print("Downloading PTB test data...")
#             try:
#                 urllib.request.urlretrieve(url, "data/ptb_test.txt")
#                 print("PTB test data downloaded successfully!")
#             except Exception as e:
#                 print(f"Download failed: {e}")
#                 print("Falling back to WikiText-2...")
#                 # Fallback to WikiText
#                 dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#                 text = "\n\n".join([item for item in dataset["text"] if item.strip()])
#                 return text
        
#         # Read the file
#         with open("data/ptb_test.txt", "r", encoding='utf-8') as f:
#             text = f.read()
        
#         return text
    
#     elif isinstance(dataset_arg, tuple):  # For Wikitext and Wikitext103
#         dataset = load_dataset(dataset_arg[0], dataset_arg[1], split="test")
#         #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
#         #dataset = load_dataset("text", data_files={"test": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1/wiki.test.raw"}, split="test")
#         #dataset = load_dataset("text",data_files={"test": "wiki.test.raw"},split="test")
#         #text = " ".join(dataset["text"])
#         text = "\n\n".join(dataset["text"])
#         return text
    
#     else: # tiny_shakespeare
#         url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#         response = requests.get(url)
#         text = response.text
#         return text



def tokenization_and_chunk(text, tokenizer, chunk_params, max_length):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    block_size = min(chunk_params.block_size, max_length)  # Ensure block_size doesn't exceed model's max
    stride = int(chunk_params.stride_ratio * block_size)  # Calculate stride based on ratio

    seq_len = tokens.size(1)
    samples = []
    begin_locs = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + block_size, seq_len)
        chunk = tokens[:, begin_loc:end_loc]
        
        # Skip chunks that are too short (less than 2 tokens)
        if chunk.size(1) < 2:
            continue
            
        samples.append(chunk)
        begin_locs.append((begin_loc, end_loc, prev_end_loc))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return samples, begin_locs

def evaluate_model(samples, begin_locs, model, device):
    nll_sum = 0.0
    n_tokens = 0
    losses = []

    print("Evaluating...")
    for i, (input_ids, (begin_loc, end_loc, prev_end_loc)) in enumerate(zip(samples, begin_locs)):
        # Keep as LongTensor for embedding layer
        input_ids = input_ids.to(device=device, dtype=torch.long)
        
        # Skip if sequence is too short
        if input_ids.size(1) < 2:
            print(f"Chunk {i+1}: Skipped (too short)")
            continue
            
        trg_len = end_loc - prev_end_loc
        target_ids = input_ids.clone()
        
        if trg_len < input_ids.size(1):
            target_ids[:, :-trg_len] = -100
        
        try:
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

            losses.append(neg_log_likelihood.item())
            #print(f"Chunk {i+1}: Loss = {neg_log_likelihood.item():.4f}, Tokens = {num_loss_tokens}")
            
        except Exception as e:
            print(f"Chunk {i+1}: Error - {str(e)}")
            print(f"Input shape: {input_ids.shape}")
            continue

    avg_nll = nll_sum / n_tokens if n_tokens > 0 else float('inf')
    # Fix: Use clone().detach() instead of torch.tensor()
    perplexity = torch.exp(avg_nll.clone().detach()).item() if n_tokens > 0 else float('inf')
    return avg_nll, perplexity, losses

def plot_loss_curve(all_losses, dataset_name):
    if not all_losses:
        print("No losses to plot.")
        return
    plt.figure(figsize=(10, 5))
    for model_name, params, losses in all_losses:
        stride = int(params.block_size * params.stride_ratio)
        plt.plot(losses, 
                label=f"{model_name} (block={params.block_size}, "
                      f"stride_ratio={params.stride_ratio:.2f}, "
                      f"stride={stride})")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title(f"Loss Per Batch for All Runs on {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 4)
    plt.xlim(0, max(len(losses) for _, _, losses in all_losses))
    plt.show()

def load_model_and_tokenizer(specific_model_name, device):
    """Load model and tokenizer with proper configuration"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            specific_model_name,
            trust_remote_code=True
        )
        
        # Configure Flash Attention
        config = {
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            #"use_flash_attention": True,
            "device_map": "cuda",
            #"torch_dtype": torch.float16 if device.type == 'cuda' else torch.float32,
            "torch_dtype": "auto"
            }
        
        # First initialize model on CPU
        model = AutoModelForCausalLM.from_pretrained(
            specific_model_name,
            **config
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {specific_model_name}: {str(e)}")
        raise

for model_name in model_arr:
    print(f"Processing model: {model_name}")

    # Fetch the specific model info from model_args
    model_info = model_args[model_name]
    specific_model_name = model_info["name"]
    max_length = model_info["max_length"]
    
    print(f"Model max length: {max_length}")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(specific_model_name, device)
    model.eval()

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loop through dataset_arr
    for dataset_name in dataset_arr:
        print(f"Processing dataset: {dataset_name}")

        # Extract text from the dataset
        text = dataset2text(dataset_name)

        all_losses = []  # Reset for each dataset

        for params in chunk_params:
            # Skip if block_size exceeds model's max length
            if params.block_size > max_length:
                print(f"Skipping chunk parameters: block_size={params.block_size} exceeds model max_length={max_length}")
                #continue
                print(f"Using chunk parameters: block_size={max_length}, stride_ratio={params.stride_ratio} "
                  f"(stride={int(max_length * params.stride_ratio)}), batch_size={params.batch_size}")
            else:    
                print(f"Using chunk parameters: block_size={params.block_size}, stride_ratio={params.stride_ratio} "
                  f"(stride={int(params.block_size * params.stride_ratio)}), batch_size={params.batch_size}")

            samples, begin_locs = tokenization_and_chunk(text, tokenizer, params, max_length)
            
            if not samples:
                print("No valid samples generated, skipping...")
                continue
                
            avg_nll, perplexity, losses = evaluate_model(samples, begin_locs, model, device)

            print(f"Average Loss: {avg_nll:.4f}")
            print(f"Perplexity: {perplexity:.4f}")

            all_losses.append((model_name, params, losses))

        # Plot the losses of all runs for the current model and dataset
        # if all_losses:
        #     plot_loss_curve(all_losses, dataset_name)

    # Flush the model GPU memory
    del model
    torch.cuda.empty_cache()