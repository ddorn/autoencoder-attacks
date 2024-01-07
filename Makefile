figures:
	~/ai/semester-project/krsync -azP --stats testjob-0-0:/scratch/diego/semester-proj/images .


# Build the latex document automatically for instant preview and sync with the source in vim
auto:
	cd tex && latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out -pvc main.tex

# Build the latex document
build:
	cd tex && latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out main.tex

# Clean the latex build files
clean:
	cd tex && latexmk -c -output-directory=out
