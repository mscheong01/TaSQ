import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    print('here1')
    try:
        print('here2')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True, batched=True)
    except:
        print('here3')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True, batched=True)
    print('here4')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    print('here5')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    print('here6')
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
    )

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if trainenc.input_ids.shape[1] - seqlen - 1 < 0:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] - seqlen - 1 <= 0:
            i = 0
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_gpqa(nsamples, seed, seqlen, model):
    """Mirrors quant/kvquant/datautils.py's get_gpqa -- same corpus, same windowing, kept in sync
    since Fisher (this file) and the k-means codebook (quant/) must see identical calibration data
    for a GPQA-calibrated run to be a clean single-factor swap vs the wikitext2 baseline."""
    from datasets import load_dataset
    data = load_dataset('Idavidrein/gpqa', 'gpqa_main', split='train')

    def format_example(ex):
        return f"Question: {ex['Question']}\n\nExplanation: {ex['Explanation']}\n\nAnswer: {ex['Correct Answer']}"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(format_example(ex) for ex in data), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_code(nsamples, seed, seqlen, model, n_files=300):
    """Mirrors quant/kvquant/datautils.py's get_code -- same corpus, same windowing, kept in sync
    for the same reason get_gpqa is: Fisher (this file) and the codebook (quant/) must see
    identical calibration data."""
    from datasets import load_dataset
    ds = load_dataset('codeparrot/codeparrot-clean', streaming=True, split='train')
    texts = []
    for i, ex in enumerate(ds):
        if i >= n_files:
            break
        texts.append(ex['content'])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_gpqa_code(nsamples, seed, seqlen, model, mix_code_n=4):
    n_gpqa = max(0, nsamples - mix_code_n)
    gpqa_loader, testenc = get_gpqa(n_gpqa, seed, seqlen, model)
    code_loader, _ = get_code(mix_code_n, seed, seqlen, model)
    return gpqa_loader + code_loader, testenc


def get_mixed(nsamples, seed, seqlen, model):
    wikiloader, _ = get_wikitext2(nsamples, seed, seqlen, model)
    c4loader, _ = get_c4(nsamples, seed, seqlen, model)
    new_loader = []
    for i in range(nsamples):
        if i%2 == 0:
            new_loader.append(wikiloader[i//2])
        else:
            new_loader.append(c4loader[i//2])
    return new_loader, None


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', mix_code_n=4
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if 'mixed' in name:
        return get_mixed(nsamples, seed, seqlen, model)
    if 'gpqa_code' in name:
        return get_gpqa_code(nsamples, seed, seqlen, model, mix_code_n=mix_code_n)
    if 'gpqa' in name:
        return get_gpqa(nsamples, seed, seqlen, model)


if __name__ == "__main__":
    loader, testloader = get_mixed(128, 0, 2048, "meta-llama/Llama-2-7b-hf")
    print(len(loader))
