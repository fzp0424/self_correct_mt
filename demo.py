from ter_lib import generate_ans, TER


def demo():
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
    T = TER(
        lang_pair=lang_pair,
        model=model,
        module="translate",
        strategy=translate_strategy,
    )
    E = TER(
        lang_pair=lang_pair, model=model, module="estimate", strategy=estimate_strategy
    )
    R = TER(lang_pair=lang_pair, model=model, module="refine", strategy=refine_strategy)

    # Load examples and set up the parser
    examples = (
        T.load_examples()
    )  # if few-shot translate is not supported, automatically use zero-shot translate
    json_output_parser = T.get_output_parser()

    # Translate
    T_messages = T.get_hydrated_prompt(
        src_lang,
        tgt_lang,
        src_text,
        json_output_parser.get_format_instructions(),
        examples,
    )
    hyp = generate_ans(model, "translate", T_messages, json_output_parser)

    # Estimate
    json_output_parser = E.get_output_parser()
    E_messages = E.get_hydrated_prompt(
        src_lang,
        tgt_lang,
        src_text,
        json_output_parser.get_format_instructions(),
        examples,
        hyp,
    )
    mqm_info, nc = generate_ans(model, "estimate", E_messages, json_output_parser)

    # Refine if necessary
    if nc == 1:
        json_output_parser = R.get_output_parser()
        R_messages = R.get_hydrated_prompt(
            src_lang,
            tgt_lang,
            src_text,
            json_output_parser.get_format_instructions(),
            examples,
            hyp,
            mqm_info,
        )
        cor = generate_ans(model, "refine", R_messages, json_output_parser)
    elif nc == 0:
        cor = hyp

    # Display translation results
    print(
        f"----------------(╹ڡ╹ )-----------Result---------o(*￣▽￣*)ブ-----------------"
    )
    print(f"Model: {model}")
    print(f"Source: {src_text}")
    print(f"Hypothesis: {hyp}")
    print(f"Correction: {cor}")
    print(f"Need correction: {nc}")
    print(f"MQM Info: {mqm_info}")


if __name__ == "__main__":
    demo()
