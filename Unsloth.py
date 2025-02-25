from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = False

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="../DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


prompt = """请写出一个恰当的回答来完成当前对话任务。

### Instruction:
你是一名助人为乐的助手。

### Question:
{}

### Response:
<think>{}"""

question="你是谁?"
EOS_TOKEN = tokenizer.eos_token
print(EOS_TOKEN)

inputs = tokenizer([prompt.format(question,"")],return_tensors="pt").to("cuda")

output = model.generate(
    input_ids=inputs.input_ids,
    max_new_tokens=1200,
    use_cache=True,
)

responses=tokenizer.batch_decode(output)
print(responses[0].split("### Response:")[1])