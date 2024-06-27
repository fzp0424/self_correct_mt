import json
import argparse
from dotenv import load_dotenv, find_dotenv
from ter_lib import generate_ans, TEaR


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser('Command-line script to use TER')
    parser.add_argument('-l', '--lang', type=str, default='zh-en', help='language pair - zh-en, fr-de, en-ru ...')
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo', help='the model endpoint used for evaluation')
    parser.add_argument('-sl', '--src_lang', type=str, default='Chinese', help='source language - Chinese, French, English ...')
    parser.add_argument('-tl', '--tgt_lang', type=str, default='English', help='target language - English, German, Russian ...')
    parser.add_argument('-ts', '--translate_strategy', type=str, default='few-shot', help='which prompting strategy is used in translating')
    parser.add_argument('-es', '--estimate_strategy', type=str, default='few-shot', help='which prompting strategy is used in estimating')
    parser.add_argument('-rs', '--refine_strategy', type=str, default='beta', help='which prompting strategy is used in refining')
    args = parser.parse_args()

    src_lan = args.src_lang
    tgt_lan = args.tgt_lang

    print(f"Please enter the {src_lan} source text to be translated:")
    src_text = input()

    # Check if args.lang exists in the JSON file using assert
    lan_pairs = read_json('language_pair.json')
    found_pair = None
    for obj in lan_pairs:
        if list(obj.keys())[0] == args.lang:
            found_pair = obj
            break

    if found_pair is None: 
        print(f"Not support few-shot {args.lang} translate. Use zero-shot translate.")
        args.translate_strategy = 'zero-shot'
        args.refine_strategy = 'alpha'

    # Initialize TER instances
    T = TEaR(lang_pair=args.lang, model=args.model, module='translate', strategy=args.translate_strategy)
    E = TEaR(lang_pair=args.lang, model=args.model, module='estimate', strategy=args.estimate_strategy)
    R = TEaR(lang_pair=args.lang, model=args.model, module='refine', strategy=args.refine_strategy)

    # Load examples and set parser
    examples = T.load_examples()
    json_parser, json_output_instructions = T.set_parser()

    # Translate
    T_messages = T.fill_prompt(src_lan, tgt_lan, src_text, json_output_instructions, examples)
    hyp = generate_ans(args.model, 'translate', T_messages, json_parser)

    # Estimate
    json_parser, json_output_instructions = E.set_parser()
    E_messages = E.fill_prompt(src_lan, tgt_lan, src_text, json_output_instructions, examples, hyp)
    mqm_info, nc = generate_ans(args.model, 'estimate', E_messages, json_parser)

    # Refine if necessary
    if nc == 1:
        json_parser, json_output_instructions = R.set_parser()
        R_messages = R.fill_prompt(src_lan, tgt_lan, src_text, json_output_instructions, examples, hyp, mqm_info)
        cor = generate_ans(args.model, 'refine', R_messages, json_parser)
    elif nc == 0:
        cor = hyp

    # Output the translation
    print(f"----------------(╹ڡ╹ )-----------TEaR---------o(*￣▽￣*)ブ-----------------")
    print(f"Model:{args.model}")
    print(f"Source: {src_text}")
    print(f"Hypothesis: {hyp}")
    print(f"Correction: {cor}")
    print(f"Need correction: {nc}")
    print(f"MQM Info: {mqm_info}")

if __name__ == '__main__':
    main()
