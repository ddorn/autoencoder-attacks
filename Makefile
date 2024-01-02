figures:
	~/ai/semester-project/krsync -azP --stats testjob-0-0:/scratch/diego/semester-proj/images .


# Build the latex document automatically for instant preview and sync with the source in vim
auto:
	latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out -pvc projet.tex

# Build the latex document
build:
	latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out projet.tex

# Clean the latex build files
clean:
	latexmk -c -output-directory=out
