"""This script parses computation logs and writes out LaTeX tabular
with results to stdout.
"""
from __future__ import print_function
import glob, os, warnings


# Directory with logs
prefix = "results"

# Snippets for LaTeX tabular
prolog = r"""\begin{table}[b]
\centering
\begin{tabular}{lrrrrrr}
%
case
    &   \#cells & $C_{\mathrm{cont}, \mathrm{PF}}$
                              & $\norm{{\cal E}_\mathrm{glob}}_q$
                                          & $\cred{N^\fq}\norm{{\cal E}_\mathrm{loc}}_q$
                                                      & $\mathrm{Eff_{\eqref{eq_lift_norm_equiv_1}}}$
                                                                  & $\mathrm{Eff_{\eqref{eq_lift_norm_equiv_2}}}$
\\\hline\hline
%
"""
mrow_begin = r"\multirow{%s}{*}{\parbox{3cm}{\centering %s $p=%s$, $N=3$}}" + os.linesep
row = r"& %s & %s & %s & %s & %s & %s \\" + os.linesep
mrow_end = r"\hline" + os.linesep
epilog = r"""\end{tabular}
\caption{Quantities of localization inequalities~\eqref{eq_lift_norm_equiv}
         (approximating~\eqref{eq_loc_dual_gal_2}, \eqref{eq_loc_dual_gal_1})
         for chosen model problems.}
\label{tab_loc}
\end{table}
"""

# Read line tagged with 'RESULT' from logs
results = {}
logs = glob.glob(os.path.join(prefix, "*.log"))
for log in logs:
    f = open(log, 'r')
    lines = f.readlines()
    l = [l for l in lines if l[:6]=="RESULT"]
    assert(len(l) in [0, 1])
    if len(l) == 0:
        warnings.warn("There is no 'RESULT' line in file '%s'!" % log)
        continue
    l = l[0]
    l = l[6:]
    l = l.split()
    key = (l[0], l[1])
    val = l[2:]
    if results.has_key(key):
        vals = results[key]
    else:
        vals = results[key] = []
    vals.append(tuple(val))

# Adjust slightly scaling of local lifting column
prolog = prolog.replace(r"\cred{N^\fq}", "")
for k in results.keys():
    for i in range(len(results[k])):
        val = float(results[k][i][3])
        N = 3
        p = float(k[1])
        q = p/(p-1.0)
        val *= N**(-1.0/q)
        old_row = results[k][i]
        new_row = old_row[:3]+('%.4f'%val,)+old_row[4:]
        results[k][i] = new_row

# Compile tabular code using results
output = prolog
for key in results.keys():
    vals = results[key]
    vals.sort(lambda x,y: 1 if int(x[0]) > int(y[0]) else -1)
    output += mrow_begin % (len(vals), key[0], key[1])
    for val in vals:
        output += row % val
    output += mrow_end
output += epilog

# Write out to stdout
print(output)
