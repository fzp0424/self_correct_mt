# TEaR Readme

# Self-Refine Machine Translation Agent - TEaR

![TEaR.png](asset/TEaR-github.png)

## **🔔 News**

- **[06/27/2024] We have updated our code, paper, and dependencies.  Add support to the glm-4-0520 and glm-4-air (from Zhipu AI). Check the latest version of our TEaR paper on arXiv.**
- **[06/22/2024] Welcome to try our newest learning-based translation-refinement tools - Ladder** https://github.com/fzp0424/Ladder**, paper link:** [[2406.15741] Ladder: A Model-Agnostic Framework Boosting LLM-based Machine Translation to the Next Level (arxiv.org)](https://arxiv.org/abs/2406.15741)
- **[02/28/2024] Our Code for TEaR is open-sourced!**
- **[02/27/2024] Our API-based Self-Refine framework TEaR is released on arXiv:** [[2402.16379] Improving LLM-based Machine Translation with Systematic Self-Correction (arxiv.org)](https://arxiv.org/abs/2402.16379)

## **🚀 Quick Links**

- **[About TEaR](#about)**
- **[Code Structure](#code)**
- **[Requirements](#req)**
- **[Usage](#us)**
- **[Citation](#cita)**

## **🤖** About TEaR<a name="about"></a>

The **TEaR** (**Translate**, **Estimate**, **a**nd **Refine**) framework is designed as a self-correcting translation agent that leverages Large Language Models (LLMs) to refine its original translation. It comprises three integral modules:

1. **Translate**: This component employs an LLM to generate the initial translation. It ensures that the translations are internally sourced, thereby maintaining a controlled environment for the translation process.
2. **Estimate**: Following the initial translation, this module takes over to evaluate the quality of the translation. It provides an assessment that acts as feedback, indicating areas of strength and potential improvement in the translation.
3. **Refine**: Based on the feedback and assessments provided by the Estimate module, the Refine module performs corrections on the initial translation. It utilizes the insights gained from the previous two modules to enhance the overall quality of the translation.

![framework.png](asset/framework.png)

## **📚** Code Structure<a name="code"></a>

- `.env`: environment file (set API keys first!!)
- `prompts/`: folder that contains all prompt files
- `dataset/`: folder that contains all data used
- `eval/`: folder that contains the code for evaluation
- `ter_lib.py`: tools, TEaR modules, etc.
- `run_file.py`: run TEaR with file input
- `run_command.py`: run TEaR with command-line input
- `demo.py`: an easy-realized TEaR demo
- `language_pair.json`: language pairs supported in our paper

## **📃** Requirements<a name="req"></a>

```
openai==1.35.3
langchain_google_genai==1.0.6
langchain==0.2.5
python-dotenv==1.0.0
zhipuai
```

## **💁** Usage<a name="us"></a>

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
from ter_lib import generate_ans, TEaR

def demo():
    # Set arguments
    lang_pair = "zh-en"
    src_lang = "Chinese"
    tgt_lang = "English"
    model = "gpt-3.5-turbo" 
    translate_strategy = "few-shot"
    estimate_strategy = "few-shot"
    refine_strategy = "beta"
    src_text = "如果EMNLP录取我的工作，那么EMNLP就是世界上最棒的NLP会议！"

    # Initialize TEaR instances
    T = TEaR(lang_pair=lang_pair, model=model, module='translate', strategy=translate_strategy)
    E = TEaR(lang_pair=lang_pair, model=model, module='estimate', strategy=estimate_strategy)
    R = TEaR(lang_pair=lang_pair, model=model, module='refine', strategy=refine_strategy)

    # Load examples and set up the parser
    examples = T.load_examples() # if few-shot translate is not supported, automatically use zero-shot translate
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
    print(f"----------------(╹ڡ╹ )----------TEaR---------o(*￣▽￣*)ブ-----------------")
    print(f"Model: {model}")
    print(f"Source: {src_text}")
    print(f"Hypothesis: {hyp}")
    print(f"Correction: {cor}")
    print(f"Need correction: {nc}")
    print(f"MQM Info: {mqm_info}")

if __name__ == '__main__':
    demo()
```

### d) Evaluation

We provide our evaluation code using `zh-en` as an example.

```bash
cd eval
python evaluation.py
```

## Citation<a name="cita"></a>

```latex
@article{feng2024improving,
  title={Improving llm-based machine translation with systematic self-correction},
  author={Feng, Zhaopeng and Zhang, Yan and Li, Hao and Liu, Wenqiang and Lang, Jun and Feng, Yang and Wu, Jian and Liu, Zuozhu},
  journal={arXiv preprint arXiv:2402.16379},
  year={2024}
}
```