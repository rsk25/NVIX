from pathlib import Path
from common import dataset
import torch
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model.swan import SWANPhase1Only
from model.base import chkpt
from test_model import load_config, run_model_for_attention
from common.dataset import Dataset
from learner import *


# 'Sears tower' problem
SPLIT = 'dev'
PROBLEM_IDX = 49
BATCH_SIZE = 1


def set_ckpt_path(dir: str, from_pretrained: bool=True):
    if from_pretrained:
        return Path('.') / 'resource' / 'runs_copy' / dir
    else:
        return Path('.') / 'resource' / dir


def display_attention(sentence: List[str], explanation: Dict[str, List[str]], \
                        attention: torch.Tensor, copy_probs: torch.Tensor, n_cols=1):
    n_rows = len(explanation)
    assert n_rows * n_cols == len(explanation)

    _attention = attention.squeeze()
    _copy_probs = copy_probs.squeeze()

    fig = plt.figure(figsize=(40, 40))

    begin_idx = 0
    end_idx = 0
    for i, k in enumerate(explanation):
        expl_list = explanation[k][0].split()+['<EOS>']
        end_idx = begin_idx + len(expl_list)
        
        # attention heatmap
        ax1 = fig.add_subplot(n_rows, n_cols*2, i*2+1) 

        _attn_for_expl = _attention[begin_idx:end_idx, :]
        _attn_for_expl = _attn_for_expl.numpy()

        cax1 = ax1.matshow(_attn_for_expl, cmap='bone')
        fig.colorbar(cax1, location='bottom')

        ax1.tick_params(labelsize=12)
        ax1.set_xticklabels(['']+['<SOS>']+[t for t in sentence]+['<EOS>'], 
                           rotation=45)
        ax1.set_yticklabels(['']+expl_list)
        ax1.set_title(k)
        ax1.set_xlabel('Source Sentence')
        ax1.xaxis.set_label_position('top')
        ax1.set_ylabel(f'Explanation for {k}')

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # copy probability heatmap
        ax2 = fig.add_subplot(n_rows, n_cols*2, i*2+2)

        _probs_for_expl = _copy_probs[begin_idx:end_idx]
        _probs_for_expl = _probs_for_expl.unsqueeze(0).numpy()

        cax2 = ax2.matshow(_probs_for_expl, cmap='bone')
        fig.colorbar(cax2, location='bottom')

        ax2.tick_params(labelsize=12)
        ax2.set_xticklabels(['']+expl_list)
        ax2.set_title(k)
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel(f'Explanation for {k}')

        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

        begin_idx = end_idx

    plt.show()
    plt.close()


if __name__ == '__main__':
    chpt_path = set_ckpt_path('best_SWAN_P1', from_pretrained=True)

    # load pretrained
    tokenizer = torch.load(chpt_path / 'tokenizer.pt')
    checkpoint = torch.load(chpt_path / 'SWANPhase1Only.pt')
    config = load_config(chpt_path)

    # set seed
    set_seed(config['seed'])

    # create model instance
    nvix = SWANPhase1Only.create_or_load(path=str(chpt_path), **config)
    nvix.eval()

    # load dataset
    dataset_path = set_ckpt_path('dataset', from_pretrained=False)
    test_data = Dataset(dataset_path / 'pen.json', number_window=3)
    test_data.select_items_with_file(dataset_path / 'experiments' / 'pen' / SPLIT)

    # load (single) batch
    batch = test_data.get_minibatches(batch_size=BATCH_SIZE, for_testing= True)
    
    # get outputs from model
    output = nvix(
        text=batch[PROBLEM_IDX].text.to(nvix.device),
        beam=3,
        beam_expl=1
    )
    explanations = output['explanation'][0].to_human_readable(tokenizer=tokenizer)

    display_attention(
        sentence=batch[PROBLEM_IDX].text.raw[0].split(),
        explanation=explanations,
        attention=nvix.attention_scores,
        copy_probs=nvix.copy_probabilities
    )
