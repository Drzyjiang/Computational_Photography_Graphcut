# CS598 Final Project Spring 23 <br>
zjiang2@illinois.edu <br>

### Reproduce of paper:
[Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources](https://dl.acm.org/doi/10.1145/3308558.3313485) <br>
WWW '19: The World Wide Web Conference, Pages 72–82

### Instructions:
1. Download MIMIC-III from PhysioNet. Only need NOTEEVENTS.csv and DIAGNOSES_ICD.csv
2. Download this notebook and three utils files (utils_preprocessing.py, utils_nn.py, utils_ksi.py)
3. Optional: Pretrained models (weights) are available at this [this link](https://drive.google.com/drive/folders/1331SQDUL_ZDvwec0IeSkSHpiRfC7u8se?usp=sharing) 
4. In Part 0, change 'dir_path' to actual directory path, and change 'device' to appropriate device
   To use the pretrained models, set 'load_pretrain' as True
5. Run Final project.ipynb as Jupyter Notebook

### Code structures:
This notebook has seven parts: <br>
Part 0: Path and device <br>
Part I: Libraries <br>
Part II: Data Preprocessing <br>
Part III: Logistic Regression (bag-of-words) <br>
Part IV: LR without and with KSI <br>
Part V: RNN without and with KSI <br>
Part VI: KSI Interpretation <br>
Part VII: Ablation study
