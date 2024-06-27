import json
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from ter_lib import generate_ans, TEaR
# Load environment variables
load_dotenv(find_dotenv())

# Helper functions
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    if os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8") as jsonfile:
            json_data = json.load(jsonfile)
    else:
        json_data = []

    json_entry = {
        "id": data['id'],
        "src": data['src'],
        "ref": data['ref'],
        "hyp": data['hyp'],
        "cor": data['cor'],
        "need correction": data['need correction'],
        "mqm_info": data['mqm_info']
    }

    json_data.append(json_entry)

    with open(path, "w", encoding="utf-8") as jsonfile:
        json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

    print(f"Successfully appended data to file {path}")
    print(f"----------------(╹ڡ╹ )-----------End---------o(*￣▽￣*)ブ-----------------")
    print(f"\n")
    return

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def main():
    # Argument parsing
    parser = argparse.ArgumentParser('Command-line script to use TEaR')
    parser.add_argument('-l', '--lang', type=str, default='zh-en', help='language pair - zhen, ende, enru')
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo', help='the model endpoint used for evaluation')
    parser.add_argument('-ts', '--translate_strategy', type=str, default='few-shot', help='which prompting strategy is used in translating')
    parser.add_argument('-es', '--estimate_strategy', type=str, default='few-shot', help='which prompting strategy is used in estimating')
    parser.add_argument('-rs', '--refine_strategy', type=str, default='beta', help='which prompting strategy is used in refining')
    parser.add_argument('--ref', type=str, default='', help='reference path')
    parser.add_argument('--src', type=str, default='', help='source text path')
    parser.add_argument('--output', type=str, default='', help='hypothesis text path')
    args = parser.parse_args()

    # Check if args.lang exists in the JSON file using assert
    lan_pairs = read_json('language_pair.json')
    found_pair = None
    for obj in lan_pairs:
        if list(obj.keys())[0] == args.lang:
            found_pair = obj
            break
    assert found_pair is not None, f"Not support {args.lang}"

    src_lan = found_pair[args.lang][0]
    tgt_lan = found_pair[args.lang][1]
    args.src = f"dataset/mt/baseline/{found_pair[args.lang][2]}{found_pair[args.lang][3]}/rand_200_test.{args.lang}.{found_pair[args.lang][2]}"
    args.ref = f"dataset/mt/baseline/{found_pair[args.lang][2]}{found_pair[args.lang][3]}/rand_200_test.{args.lang}.{found_pair[args.lang][3]}"
    args.output = f"result/{args.model}_{args.lang}_{args.translate_strategy}_{args.estimate_strategy}_{args.refine_strategy}.json"

    if os.path.isfile(args.output):
        if os.path.getsize(args.output) > 0:
            with open(args.output, "r", encoding="utf-8") as jsonfile:
                json_data = json.load(jsonfile)
        else:
            json_data = []
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as jsonfile:
            json_data = []

    print(f"Have translated {len(json_data)} segments!")


    # Initialize TER instances
    T = TEaR(lang_pair=args.lang, model=args.model, module='translate', strategy=args.translate_strategy)
    E = TEaR(lang_pair=args.lang, model=args.model, module='estimate', strategy=args.estimate_strategy)
    R = TEaR(lang_pair=args.lang, model=args.model, module='refine', strategy=args.refine_strategy)

    # Read source and reference texts
    srcs = read_txt(args.src)
    refs = read_txt(args.ref)
    print(f"Loaded {len(srcs)} source segments!")
    assert len(srcs) == len(refs), "please check src and ref files"

    for index, line in enumerate(srcs):
        write_in_json = {}

        if os.path.getsize(args.output) > 0:
            with open(args.output, "r", encoding="utf-8") as jsonfile:
                json_data = json.load(jsonfile)
        else:
            json_data = []

        # Check if index already exists in JSON data
        existing_ids = [obj["id"] for obj in json_data]
        if index in existing_ids:
            continue

        print(f"----------------(╹ڡ╹ )---------Begin {index}-------o(*￣▽￣*)ブ----------------")
        # Load examples and set parser
        examples = T.load_examples()
        json_parser, json_output_instructions = T.set_parser()

        # Translate
        T_messages = T.fill_prompt(src_lan, tgt_lan, line, json_output_instructions, examples)
        hyp = generate_ans(args.model, 'translate', T_messages, json_parser)

        # Estimate
        json_parser, json_output_instructions = E.set_parser()
        E_messages = E.fill_prompt(src_lan, tgt_lan, line, json_output_instructions, examples, hyp)
        mqm_info, nc = generate_ans(args.model, 'estimate', E_messages, json_parser)

        # Refine if necessary
        if nc == 1:
            json_parser, json_output_instructions = R.set_parser()
            R_messages = R.fill_prompt(src_lan, tgt_lan, line, json_output_instructions, examples, hyp, mqm_info)
            cor = generate_ans(args.model, 'refine', R_messages, json_parser)
        elif nc == 0:
            cor = hyp

        # Prepare data for writing to JSON
        write_in_json['id'] = index
        write_in_json['src'] = srcs[index]
        write_in_json['ref'] = refs[index]
        write_in_json['hyp'] = hyp
        write_in_json['cor'] = cor
        write_in_json['need correction'] = nc
        write_in_json['mqm_info'] = mqm_info

        # Save to JSON file
        save_json(write_in_json, args.output)

if __name__ == '__main__':
    main()
