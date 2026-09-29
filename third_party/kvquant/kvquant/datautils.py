import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

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

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
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
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
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
            if trainenc.input_ids.shape[1] > seqlen:
                break
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
            if tmp.input_ids.shape[1] > seqlen:
                break
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
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
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
            if trainenc.input_ids.shape[1] > seqlen:
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
    """Sample calibration windows from the gated ``gpqa_main`` split.

    Examples are concatenated before windowing. The returned test encoding uses WikiText-2 and
    is not used by this project's perplexity path.
    """
    from datasets import load_dataset
    data = load_dataset('Idavidrein/gpqa', 'gpqa_main', split='train')

    def format_example(ex):
        return f"Question: {ex['Question']}\n\nExplanation: {ex['Explanation']}\n\nAnswer: {ex['Correct Answer']}"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(format_example(ex) for ex in data), return_tensors='pt')

    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

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

def get_code(nsamples, seed, seqlen, model, n_files=300):
    """Sample calibration windows from ``codeparrot/codeparrot-clean``.

    The first ``n_files`` files are streamed deterministically and concatenated before windowing.
    Evaluation datasets such as HumanEval and MBPP are not used for calibration.
    """
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
    """Mix ``nsamples - mix_code_n`` GPQA windows with ``mix_code_n`` code windows."""
    n_gpqa = max(0, nsamples - mix_code_n)
    gpqa_loader, testenc = get_gpqa(n_gpqa, seed, seqlen, model)
    code_loader, _ = get_code(mix_code_n, seed, seqlen, model)
    return gpqa_loader + code_loader, testenc


def get_mixed(nsamples, seed, seqlen, model, mix_c4_n=4):
    """wikitext2 calibration with c4 samples mixed in.

    Addresses a comma-coverage hole in wikitext2: the "," token appears there as " , " and in c4
    as ",", so a codebook calibrated on wikitext2 alone sees only one of the two forms. The fix is
    applied to the calibration set rather than by reweighting the codebook -- nsamples-mix_c4_n
    windows from wikitext2, mix_c4_n from c4, concatenated.

    testenc is wikitext2's, unchanged and unused here: PPL evaluation in this project is a
    separate script.
    """
    n_wiki = max(0, nsamples - mix_c4_n)
    wiki_loader, testenc = get_wikitext2(n_wiki, seed, seqlen, model)
    c4_loader, _ = get_c4(mix_c4_n, seed, seqlen, model)
    return wiki_loader + c4_loader, testenc


def get_gpqa_diamond_windowed(nsamples, seed, seqlen, model):
    """llama_simquant.py-SPECIFIC variant: its own calibration/k-means-fitting pipeline
    (llama_calibration()) preallocates a FIXED (nsamples, seqlen, hidden) tensor and assigns
    each captured batch into one row -- it cannot accept GPQA-Diamond's true variable-length
    prompts (148-2818 tokens; confirmed by a real RuntimeError: "expanded size...must match...
    Target sizes: [2048,4096]. Tensor sizes: [148,4096]"). make_tasq.py/build_nova_k.py/
    run-fisher-my.py have no such fixed-shape buffer and correctly use the TRUE per-prompt
    recipe (get_gpqa_diamond below) -- only THIS script's k-means-training-sample construction
    needs windowing, same "concatenate then random-window" pattern get_gpqa/get_gpqa_code
    already use, just sourced from gpqa_diamond instead of gpqa_main."""
    from datasets import load_dataset
    data = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')

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


def get_gpqa_diamond(nsamples, seed, seqlen, model):
    """Build NovaKV calibration inputs from individual GPQA-Diamond prompts.

    `seed` is retained for loader compatibility; `seqlen` bounds each prompt's token length.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    data = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train')
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    GPQA_TMPL = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}"
    )
    n = nsamples if nsamples and nsamples > 0 else len(data)
    trainloader = []
    for ex in list(data)[:n]:
        p = GPQA_TMPL.format(Question=ex['Question'], A=ex['Correct Answer'],
                              B=ex['Incorrect Answer 1'], C=ex['Incorrect Answer 2'],
                              D=ex['Incorrect Answer 3'])
        text = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                              add_generation_prompt=True, tokenize=False)
        inp = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids
        if seqlen:
            inp = inp[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', mix_c4_n=4
):
    if 'mixed' in name:
        return get_mixed(nsamples, seed, seqlen, model, mix_c4_n=mix_c4_n)
    if 'gpqa_diamond' in name:
        return get_gpqa_diamond_windowed(nsamples, seed, seqlen, model)
    if 'gpqa_code' in name:
        return get_gpqa_code(nsamples, seed, seqlen, model, mix_code_n=mix_c4_n)
    if 'gpqa' in name:
        return get_gpqa(nsamples, seed, seqlen, model)
    if name == 'code':
        return get_code(nsamples, seed, seqlen, model)
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
