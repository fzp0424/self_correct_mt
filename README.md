# Self-Correct Machine Translation

![f1(1).png](Self-Correct%20Machine%20Translation%2011f9e63afc4d4b60a95b63fb09e2dcf7/f1(1).png)

## **🔔 News**

- **[02/28/2024] Our Code for TER is open-sourced!**
- **[02/27/2024] Our API-based Self-Correct framework TER is released on arXiv:** [[2402.16379] Improving LLM-based Machine Translation with Systematic Self-Correction (arxiv.org)](https://arxiv.org/abs/2402.16379)

## **🚀 Quick Links**

- [About TER](https://www.notion.so/Self-Correct-Machine-Translation-11f9e63afc4d4b60a95b63fb09e2dcf7?pvs=21)
- [Code Structure](https://www.notion.so/Self-Correct-Machine-Translation-11f9e63afc4d4b60a95b63fb09e2dcf7?pvs=21)
- [Requirements](https://www.notion.so/Self-Correct-Machine-Translation-11f9e63afc4d4b60a95b63fb09e2dcf7?pvs=21)
- [Usage](https://www.notion.so/Self-Correct-Machine-Translation-11f9e63afc4d4b60a95b63fb09e2dcf7?pvs=21)
- [Citation](https://www.notion.so/Self-Correct-Machine-Translation-11f9e63afc4d4b60a95b63fb09e2dcf7?pvs=21)

## **🤖** About TER

The **TER** (**Translate**, **Estimate**, and **Refine**) framework is designed as a self-correcting translation approach that leverages Large Language Models (LLMs) to enhance translation quality. It comprises three integral modules:

1. **Translate**: This component employs an LLM to generate the initial translation. It ensures that the translations are internally sourced, thereby maintaining a controlled environment for the translation process.
2. **Estimate**: Following the initial translation, this module takes over to evaluate the quality of the translation. It provides an assessment that acts as feedback, indicating areas of strength and potential improvement in the translation.
3. **Refine**: Based on the feedback and assessments provided by the Estimate module, the Refine module performs corrections on the initial translation. It utilizes the insights gained from the previous two modules to enhance the overall quality of the translation.

![framework.png](Self-Correct%20Machine%20Translation%2011f9e63afc4d4b60a95b63fb09e2dcf7/framework.png)

## **📚** Code Structure

- `.env`: environment file (set API keys first!!)
- `prompts/`: folder that contains all prompt files
- `dataset/`: folder that contains all data used
- `ter_lib.py`: tools, TER modules, etc.
- `run_file.py`: run TER with file input
- `run_command.py`: run TER with command-line input
- `demo.py`: an easy-realized TER demo
- `language_pair.json`: language pairs supported in our paper

## **📃** Requirements

```
openai==1.6.1
langchain_google_genai==0.0.5
langchain==0.0.352
python-dotenv==1.0.0
```

## **💁** Usage

### a) File in, File out

```
python run_file.py -l zh-en -m gpt-3.5-turbo -ts few-shot -es few-shot -rs beta
```

### b) Input a sentence to be translated by command

```
python run_cmd.py -l zh-en -m gpt-3.5-turbo -sl Chinese -tl English -ts few-shot -es few-shot -rs beta
```

### c) Demo

```python
from ter_lib import generate_ans, TER    
# Set arguments
lang_pair = "zh-en"
src_lang = "Chinese"
tgt_lang = "English"
model = "gpt-3.5-turbo"
translate_strategy = "few-shot"
estimate_strategy = "few-shot"
refine_strategy = "beta"
src_text = "如果ACL录取我的工作，那么ACL就是世界上最棒的NLP会议，不然的话，这个结论就有待商榷。"

# Initialize TER instances
T = TER(lang_pair=lang_pair, model=model, module='translate', strategy=translate_strategy)
E = TER(lang_pair=lang_pair, model=model, module='estimate', strategy=estimate_strategy)
R = TER(lang_pair=lang_pair, model=model, module='refine', strategy=refine_strategy)

# Load examples and set up the parser
examples = T.load_examples() # If few-shot translate is not supported, automatically use zero-shot translate
json_parser, json_output_instructions = T.set_parser()

# Translate
T_messages = T.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples)
hyp = generate_ans(model, 'translate', T_messages, json_parser)

# Estimate
json_parser, json_output_instructions = E.set_parser()
E_messages = E.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples, hyp)
mqm_info, nc = generate_ans(model, 'estimate', E_messages, json_parser)

# Refine if necessary
if nc == 1:
    json_parser, json_output_instructions = R.set_parser()
    R_messages = R.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples, hyp, mqm_info)
    cor = generate_ans(model, 'refine', R_messages, json_parser)
elif nc == 0:
    cor = hyp

# Display translation results
print(f"----------------(╹ڡ╹ )-----------Result---------o(*￣▽￣*)ブ-----------------")
print(f"Model: {model}")
print(f"Source: {src_text}")
print(f"Hypothesis: {hyp}")
print(f"Correction: {cor}")
print(f"Need correction: {nc}")
print(f"MQM Info: {mqm_info}")
```

## Citation

```latex
@misc{feng2024improving,
title={Improving LLM-based Machine Translation with Systematic Self-Correction},
author={Zhaopeng Feng and Yan Zhang and Hao Li and Wenqiang Liu and Jun Lang and Yang Feng and Jian Wu and Zuozhu Liu},
year={2024},
eprint={2402.16379},
archivePrefix={arXiv},
primaryClass={[cs.CL](http://cs.cl/)}
}
```