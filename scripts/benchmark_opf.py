
import pandas as pd
import numpy as np

RESULTS_DIR = "results/casis/"

def import_benchmark(filename):
    df = pd.read_csv(filename, sep="\t", header=None, index_col=0, usecols=[0, 1, 2, 3, 4, 5, 6])
    df.columns = ["status", "it", "obj", "total", "AD", "lin"]
    df = df.astype({'it': int})
    return df



subcols = ["it", "AD", "lin", "total"]
df_hsl = import_benchmark(f"{RESULTS_DIR}/pglib-full-madnlp-hsl-ma27.csv")
# df_sckkt_cpu = import_benchmark(f"{RESULTS_DIR}/pglib-full-madnlp-sckkt-cholmod.csv")
df_sckkt_cuda = import_benchmark(f"{RESULTS_DIR}/pglib-full-madnlp-sckkt-cudss-cholesky.csv")

# df_hckkt_cpu = import_benchmark(f"{RESULTS_DIR}/pglib-full-madnlp-hckkt-cholmod-7.csv")
df_hckkt_cuda = import_benchmark(f"{RESULTS_DIR}/pglib-full-madnlp-hckkt-cudss-ldl-7.csv")

df_total = pd.concat([df[subcols] for df in [df_hsl, df_sckkt_cuda, df_hckkt_cuda]], axis=1)

print(df_total.to_latex(float_format="%.2f"))
print(df_hsl[["status"]])


