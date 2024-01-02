figures:
	~/ai/semester-project/krsync -azP --stats testjob-0-0:/scratch/diego/semester-proj/images .


# Build the latex document automatically for instant preview
auto:
	zathura out/projet.pdf &
	latexmk -pvc -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" projet.tex

# Build the latex document
build:
	latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" projet.tex -outdir=out

# Clean the latex build files
clean:
	latexmk -c -outdir=out

