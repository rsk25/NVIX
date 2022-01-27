from pathlib import Path
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_ckpt_path(dir: str, from_pretrained: bool=True):
    if from_pretrained:
        return Path('.') / 'runs_copy' / dir
    else:
        return Path('.') / 'resource' / dir


def _save_figure(figure_name: str, format: str='png'):
    file_name = Path('.') / 'figures' / f'{figure_name}.{format}'
    plt.savefig(file_name, bbox_inches='tight', facecolor='white')


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

    if figure_name is not None:
        _save_figure(figure_name)

    plt.show()
    plt.close()


__all__ = ['set_ckpt_path', 'display_heatmap']