from pathlib import Path
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict

def set_ckpt_path(dir: str, from_pretrained: bool=True):
    if from_pretrained:
        return Path('.') / 'runs_copy' / dir
    else:
        return Path('.') / 'resource' / dir


def _save_figure(figure_name: str, format: str='png'):
    file_name = Path('.') / 'figures' / f'{figure_name}.{format}'
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')


def _reorder_mapping(orig_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    new_map = OrderedDict()
    for k, v in orig_map.items():
        if k[0] == 'X':
            new_map[k] = v[0].split()
    for k, v in orig_map.items():
        if k[0] == 'N':
            new_map[k] = v[0].split()
    print(new_map)
    return new_map


def display_heatmap(sentence: List[str], 
                    explanation: Dict[str, List[str]],
                    attention: torch.Tensor, 
                    copy_probs: torch.Tensor, 
                    figure_name: str = None,
                    n_cols=1):

    n_rows = len(explanation)
    assert n_rows * n_cols == len(explanation)

    _attention = attention.squeeze()
    _copy_probs = copy_probs.squeeze()

    fig = plt.figure(figsize=(40, 40))

    new_map = _reorder_mapping(explanation)
    begin_idx = 0
    end_idx = 0
    for i, k in enumerate(new_map):
        expl_list = new_map[k]+['<EOS>'] # explanation mapping과 python이 iter하는 방법이 일치하지 않을 수도
        end_idx = begin_idx + len(expl_list)
        
        # attention heatmap
        ax1 = fig.add_subplot(n_rows, n_cols*2, i*2+1) 

        _attn_for_expl = _attention[begin_idx:end_idx, :].numpy()

        cax1 = ax1.matshow(_attn_for_expl, cmap='bone')
        fig.colorbar(cax1, location='bottom')

        ax1.tick_params(labelsize=12)
        ax1.set_xticklabels(['']+['<BOS>']+[t for t in sentence]+['<EOS>'], 
                           rotation=90)
        ax1.set_yticklabels(['']+expl_list)
        ax1.set_title(k)
        ax1.set_xlabel('Source Sentence')
        ax1.xaxis.set_label_position('top')
        ax1.set_ylabel(f'Explanation for {k}')

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # copy probability heatmap
        ax2 = fig.add_subplot(n_rows, n_cols*2, i*2+2)

        _probs_for_expl = _copy_probs[begin_idx:end_idx].unsqueeze(0).numpy()

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

    if figure_name is not None:
        _save_figure(figure_name)

    plt.show()
    plt.close()


__all__ = ['set_ckpt_path', 'display_heatmap']