import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="The csv file to chose from to create a Latex base table.",
)
args = parser.parse_args()

file = args.file
print(f"Reading values from {file}")

df = pd.read_csv(file)

df = df.filter(like="_test")

print(df)

latex_table_str = """
\\begin{table}[H]
\\caption[]{}
\\label{tab:}
\\begin{center}
    \\begin{tabular}[c]{l"""

runs_str = "\n         "

for i in range(len(df.index)):
    latex_table_str += "|c"
    runs_str += f"& {str(i)} "

latex_table_str += f"""}}\\hline
         & \\multicolumn{{{len(df.index)}}}{{c}}{{Runs}}\\\\\\hline"""

latex_table_str += runs_str + "\\\\\\hline"

for name, values in df.iteritems():
    name = name.removesuffix("_test").upper()
    if name == "AVG":
        name = "Average"

    latex_table_str += f"\n         {name} "

    for val in values:
        print(val)
        latex_table_str += f"& {val} "
    latex_table_str += "\\\\"

latex_table_str += """\n        \\hline
    \\end{tabular}
\\end{center}
\\end{table}"""

print(latex_table_str)

# \begin{table}[H]
#    \caption{}
#    \label{tab:}
#    \begin{center}
#        \begin{tabular}[c]{c|c|c|c|c}
#            \hline
#            & \multicolumn{4}{c}{Runs} \\\hline
#            Dataset & 0 & 1 & 2 & 3 \\\hline
#            CIFAR10 & 0.000 & 1.000 & 2.000 & 3.000 \\
#            \hline
#        \end{tabular}
#    \end{center}
# \end{table}
