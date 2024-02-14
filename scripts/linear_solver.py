
import sys
import numpy as np
import pandas as pd


input_file = sys.argv[1]

cols = [
    "status JuMP", "it JuMP", "obj JuMP", "AD JuMP", "total JuMP", "status Exa", "it Exa", "obj Exa", "AD Exa", "total Exa",
]
benchmark = pd.read_csv(input_file, index_col=0, header=None, sep="\t")

benchmark.columns = cols
benchmark = benchmark.astype({'status JuMP': int, 'it JuMP': int, 'status Exa': int,'it Exa': int})
# benchmark.columns = ["n", "nnz", "analysis", "factorization", "backsolve", "accuracy"]
print(benchmark.to_latex(float_format="%.2e"))
