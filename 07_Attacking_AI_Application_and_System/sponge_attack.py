from transformers import AutoTokenizer
import json
import sys

model = 'openai-community/gpt2'

# Get inputs from command line arguments or use interactive mode
if len(sys.argv) > 1:
    texts = sys.argv[1:]
else:
    texts = []
    while 1:
        text = input("> ")
        if not text:
            break
        texts.append(text)

tokenizer = AutoTokenizer.from_pretrained(model)

for text in texts:
    print(f"\n{'='*60}")
    print(f"Input: {text}")
    print(f"{'='*60}")
    tokens = tokenizer.tokenize(text)
    print(f"Number of Input Characters: {len(text)}")
    print(f"Number of Input Tokens: {len(tokens)}")
    print(json.dumps(tokens, indent=2))


    