import os
from matplotlib import rcParams


if not os.path.isdir('./figures'):
    os.makedirs('./figures')


# ICML: textwidth=487.8225, columnwidth=234.8775
# ACM: textwidth=506.295, columnwidth=241.14749
def get_figsize(columnwidth=234.8775, wf=1.0, hf=(5. ** 0.5 - 1.0) / 2.0, b_fixed_height=False):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex (pt). Get this from LaTeX
                             using \showthe\columnwidth (or \the\columnwidth)
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    if b_fixed_height:
        fig_height = hf
    return [fig_width, fig_height]


def set_rc_params():
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})
    rcParams['axes.titlesize'] = 5.8
