# FOSI

FOSI is a library for improving first order optimizers with second order information.
TODO.

For more information, see our ICML 2023 paper, [FOSI: Hybrid First and Second Order Optimization](TODO).

## Installation

FOSI is written in pure Python.
Use the following instructions to install a
binary package with `pip`, or to download FOSI's source code.
We support installing `fosi` package on Linux (Ubuntu 18.04 or later).
**The installation requires Python >=3.9, <3.11**.

To download FOSI's source code run:
```bash
git clone https://github.com/hsivan/fosi
```
Let `fosi_root` be the root folder of the project on your local computer, for example `/home/username/fosi`.

To install FOSI run:
```bash
pip install git+https://github.com/hsivan/fosi.git
```
Or, download the code and then run:
```bash
pip install <fosi_root>
```

## Usage example

The following example shows how to apply FOSI with the base optimizer Adam.
TODO

## Reproduce paper's experimental results

TOSO

## Citing AutoMon

If AutoMon has been useful for your research, and you would like to cite it in an academic
publication, please use the following Bibtex entry:
```bibtex
@inproceedings{sivan_fosi_2023,
  author    = {Sivan, Hadar and Gabel, Moshe and Schuster, Assaf},
  title     = {{FOSI}: Hybrid First and Second Order Optimization},
  year      = {2023},
  series    = {ICML '23},
  booktitle = {Proceedings of the 2022 {ICML} International Conference on Machine Learning},
  note      = {to appear}
}
```
