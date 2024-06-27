def get_chrfpp_score(hyps,refs):
    import sacrebleu
    assert len(hyps) == len(refs)
    score = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    print("chrfpp:", round((score), 2))
    return score

def get_bleu_score(hyps,refs,**kwargs,):
    import sacrebleu
    assert len(hyps) == len(refs)
    score = sacrebleu.corpus_bleu(hyps,[refs],**kwargs).score
    print("sacrebleu:", round((score), 2))
    return score

def get_comet_score(src,hyps,refs):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src": s,
            "mt": h,
            "ref": r
        }
        for s,h,r in zip(src,hyps,refs)
    ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    print("comet:", round((model_output.system_score)*100, 2))
    return model_output.system_score

def get_comet_kiwi_score(src,hyps):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src":s,
            "mt":m,
        }
        for s,m in zip(src,hyps)
    ]
    model_output = model.predict(data, batch_size=8, gpus=1)
    print("comet_kiwi:", round((model_output.system_score)*100,2))
    return model_output.system_score

def get_bleurt_score(hyps,refs,batch_size=8,device='cuda:0'):
    # pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
    import torch
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    model.eval()
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

    hyps = [hyps[idx:idx+batch_size] for idx in range(0,len(hyps),batch_size)]
    refs = [refs[idx:idx+batch_size] for idx in range(0,len(refs),batch_size)]

    ret = []
    for hyp,ref in zip(hyps,refs):
        with torch.no_grad():
            inputs = tokenizer(ref, hyp, padding='longest', return_tensors='pt', max_length=model.config.max_position_embeddings,truncation=True).to(device)
            res = model(**inputs).logits.flatten().cpu().tolist()
            ret.extend(res)
    print("bleurt:", round((sum(ret)/len(ret))*100, 2))
    return sum(ret)/len(ret)

def eval_translation(src,hyps,refs,tokenize='flores200'):
    assert len(refs)==len(hyps)==len(src),f"len(refs)={len(refs)},len(hyps)={len(hyps)},len(src)={len(src)}"

    return {
        "bleu":round(get_bleu_score(hyps,refs,tokenize=tokenize),2),
        "chrfpp":round(get_chrfpp_score(hyps,refs),2),
        "comet":round(get_comet_score(src,hyps,refs)*100,2),
        'comet_kiwi':round(get_comet_kiwi_score(src,hyps)*100,2),
        "bleurt":round(get_bleurt_score(hyps,refs)*100,2),
    }
    
    
def main():
    
    classname = "zh-en"
    src_path = f"{classname}/src.txt"
    ref_path = f"{classname}/ref.txt"
    it_path = f"{classname}/it.txt"
    tear_path = f"{classname}/TEaR.txt"

    src = open(src_path, 'r', encoding='utf-8').readlines()
    ref = open(ref_path, 'r', encoding='utf-8').readlines()
    its = open(it_path, 'r', encoding='utf-8').readlines()
    refines = open(tear_path, 'r', encoding='utf-8').readlines()

    result_hyps1 = eval_translation(src, its, ref, 'flores200')
    result_hyps2 = eval_translation(src, refines, ref, 'flores200')

    print("Initial Translation:\n", result_hyps1)
    print("TEaR:\n", result_hyps2)

if __name__ == "__main__":
    main()