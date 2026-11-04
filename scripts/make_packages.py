"""Build main_submission.tex (bbl inlined, for APS) and the arXiv folder (main_arxiv.tex with the SM appended)."""
import os, re, shutil, subprocess
L='/home/claude/paper/latex'
s=open(f'{L}/main.tex').read(); b=open(f'{L}/main.bbl').read()
assert "\\bibliography{refs}" in s
sub=s.replace("\\bibliography{refs}", "% Bibliography pasted from main.bbl (APS asks for the .bbl inside the main .tex at submission)\n"+b)
open(f'{L}/main_submission.tex','w').write(sub)
# arXiv version: same, plus the SM appended page by page (pdfpages does not work with REVTeX's output routine)
n=int(subprocess.run(['pdfinfo',f'{L}/sm.pdf'],capture_output=True,text=True).stdout.split('Pages:')[1].split()[0])
line="\\newpage\\thispagestyle{empty}\\noindent\\hspace*{-1in}\\hspace*{-\\oddsidemargin}\\smash{\\raisebox{\\dimexpr-\\paperheight+1in+\\topmargin+\\headheight+\\headsep+\\topskip\\relax}[0pt][0pt]{\\includegraphics[page=%d,width=\\paperwidth,height=\\paperheight,keepaspectratio]{sm.pdf}}}"
pages="\n".join(line % (k+1) for k in range(n))
arx=sub.replace("\\usepackage{xcolor}", "\\usepackage{xcolor}\n% arXiv version: the Supplemental Material (sm.pdf) is appended below, one page per page", 1)
arx=arx.replace("\n\\end{document}", "\n\\clearpage\\onecolumngrid\\pagestyle{empty}\n"+pages+"\n\\end{document}")
A='/home/claude/paper/arxiv'
if os.path.exists(A): shutil.rmtree(A)
os.makedirs(A+'/figures')
open(f'{A}/main_arxiv.tex','w').write(arx)
for f in ('table_mnist.tex','sm.pdf'): shutil.copy(f'{L}/{f}', A)
for f in ('fig1','fig2','fig3','fig4','fig5'): shutil.copy(f'{L}/figures/{f}.pdf', f'{A}/figures/{f}.pdf')
abs_tex=s[s.index("\\begin{abstract}")+len("\\begin{abstract}"):s.index("\\end{abstract}")].strip()
for k,v in (("\\partial_{\\sw}","\\partial_{\\sigma_w^2}"),("\\sw","\\sigma_w^2"),("\\sbb","\\sigma_b^2"),("\\qs","q^*")):
    abs_tex=abs_tex.replace(k,v)
open(f'{A}/abstract_for_arxiv_form.txt','w').write(abs_tex+"\n")
print('abstract characters:', len(abs_tex))
open(f'{A}/README.txt','w').write("""arXiv source package.
Upload main_arxiv.tex, table_mnist.tex, figures/ and sm.pdf (not this README or the abstract file). arXiv compiles main_arxiv.tex
with pdflatex; the bibliography is already inlined (no BibTeX needed) and the Supplemental Material (sm.pdf, built
from latex/sm.tex) is appended to the article page by page (graphicx). Paste abstract_for_arxiv_form.txt into the arXiv abstract field (plain TeX, no macros); primary category cs.LG (or cond-mat.dis-nn), cross-list the other.
""")
r=subprocess.run(['pdflatex','-interaction=nonstopmode','main_arxiv.tex'],cwd=A,capture_output=True,text=True)
r=subprocess.run(['pdflatex','-interaction=nonstopmode','main_arxiv.tex'],cwd=A,capture_output=True,text=True)
print('arXiv build:', 'ok' if r.returncode==0 else 'FAILED', subprocess.run(['pdfinfo',f'{A}/main_arxiv.pdf'],capture_output=True,text=True).stdout.split('Pages:')[1].split()[0], 'pages')
for f in os.listdir(A):
    if f.endswith(('.aux','.log','.out')): os.remove(f'{A}/{f}')
