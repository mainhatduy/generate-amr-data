# load config info from configs/configs.json

# step1: turn vLLM services on
# step2: loacdata from hf
# step3: foreach sample in dataset create prompt for that then send to vLLM model generate
    # foreach sample success save it into systhetic_data.jsonl folow format from schema/data_schema.py (SystheticData)
    