from ter_lib import generate_ans, TER


def demo():
    # Set arguments
    lang_pair = "en-ru"
    src_lang = "English"
    tgt_lang = "Russian"
    model = "gpt-4"
    translate_strategy = "few-shot"
    estimate_strategy = "few-shot"
    refine_strategy = "beta"
    src_text = "Use Reader Line app to guide and enhance your experience with reading ruler to maintain your focus with reading line."

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
    mqm_info, is_correction_needed = generate_ans(
        model, "estimate", E_messages, json_output_parser
    )

    # Refine if necessary
    if is_correction_needed:
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
    else:
        cor = hyp

    # Display translation results
    print(
        f"----------------(╹ڡ╹ )-----------Result---------o(*￣▽￣*)ブ-----------------"
    )
    print(f"Model: {model}")
    print(f"Source: {src_text}")
    print(f"Hypothesis: {hyp}")
    print(f"Correction: {cor}")
    print(f"Need correction: {is_correction_needed}")
    print(f"MQM Info: {mqm_info}")


if __name__ == "__main__":
    demo()
