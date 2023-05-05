# CS598 Final Project Spring 23 <br>
zjiang2@illinois.edu <br>

### Reproduce of paper:
[Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources](https://dl.acm.org/doi/10.1145/3308558.3313485)

### Instructions:
1. Download MIMIC-III from PhysioNet. Only need NOTEEVENTS.csv and DIAGNOSES_ICD.csv
2. Download this notebook and three utils files (utils_preprocessing.py, utils_nn.py, utils_ksi.py)
3. In Part 0, change 'dir_path' to actual directory path, and change 'device' to appropriate device
4. Run the notebook

### Code structures:
This notebook has seven parts: <br>
Part 0: Path and device <br>
Part I: Libraries <br>
Part II: Data Preprocessing <br>
Part III: Logistic Regression (bag-of-words) <br>
Part IV: LR without and with KSI <br>
Part V: RNN without and with KSI <br>
Part VI: KSI Interpretation <br>
