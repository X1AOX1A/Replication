import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model
checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"
assistant_checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

# overwrite _assisted_decoding
import types
from generation_utils import assisted_decoding
model._assisted_decoding = types.MethodType(assisted_decoding, model)

# input
messages = [
    {'role': 'user', 'content': 'Hello, how are you?'},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)


# generate output
outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
    return_dict_in_generate=True,
    max_length=200,
    pad_token_id=tokenizer.eos_token_id,
)

output, token_dict = outputs.sequences, outputs.token_dict
output = tokenizer.decode(output[0], skip_special_tokens=False)
output = output[len(prompt):].strip()
print(output)


def highlight_tokens(self, token_dict):
    highlight_accept = lambda text: f"<span style='color: orange;'>{text}</span>"
    highlight_reject = lambda text: f"<span style='color: gray; text-decoration: line-through;'>{text}</span>"

    # Improved decode function that handles newlines correctly
    def decode(token):
        if len(token) == 0:
            return ""
        text = self.tokenizer.decode(token[0], skip_special_tokens=False)
        # Replace newlines with HTML break tags
        text = text.replace('\n', '<br>')
        return text

    output_text = ""
    for accept_tokens, reject_tokens, next_token in zip(
        token_dict["accept_tokens"], token_dict["reject_tokens"], token_dict["next_token"]):
        output_text += highlight_accept(decode(accept_tokens))
        output_text += highlight_reject(decode(reject_tokens))
        output_text += decode(next_token)

    # Wrap in a div with white-space: pre-wrap to preserve other whitespace
    output_text = f"<div style='white-space: pre-wrap;'>{output_text}</div>"

    return output_text

text = highlight_tokens(token_dict)
print(text)