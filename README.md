# HTM effect of perovskite solar cell performance
This repository contains the database and code for **Data-driven analysis of hole-transporting materials for perovskite solar cells performance** by *M del Cueto*, *K Rawski-Furman*, *J Aragó* , *E Ortí* and *A Troisi*

---

## Prerequisites

The necessary packages (with the tested versions with Python 3.8.10) are specified in the file requirements.txt. These packages can be installed with pip:

```
pip3 install -r requirements.txt
```

---

## Contents


- **database_heterogeneous.csv**: file containing the data for all HTM for all perovskite families and device architectures

- **database_homogeneous.csv**: file containing the data of HTMs with a MAPbX3 perovskite and mesoporous architecture

- **htms_PCE_predictor.py**: code to perform the ML predictions

- **plot_fig4**: directory containing all files to reproduce Figure 4 of the manuscript

- **plot_fig6**: directory containing all files to reproduce Figure 6 of the manuscript

---

## Usage and examples

- The directories *plot_fig4* and *plot_fig6* contain the input files with the necessary information to reproduce Figure 4 and Figure 6, respectively. To do this, in each directory simply run:

```
python htms_PCE_predictor.py
```

This will generate a .png file with the results in: *Figure.png*

---

## License
**Authors**: Marcos del Cueto, Karol Rawski-Furman, Juan Aragó, Enrique Ortí and Alessandro Troisi

Licensed under the [MIT License](LICENSE)
