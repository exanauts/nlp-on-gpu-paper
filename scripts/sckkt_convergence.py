
import pandas as pd
import numpy as np

RESULTS_DIR = "results/condensed/"

result = np.loadtxt(f"{RESULTS_DIR}/sckkt-stats.txt")

df = pd.DataFrame(result.T)
df = pd.read_csv(f"{RESULTS_DIR}/sckkt-stats.txt", sep="\t", header=None)
print(df.to_latex(float_format="%.5f"))

