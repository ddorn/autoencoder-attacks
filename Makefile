KRSYNC=~/ai/semester-project/krsync

figures:
	$(KRSYNC) -azP --stats testjob-0-0:/scratch/diego/semester-proj/images .


# Build the latex document automatically for instant preview and sync with the source in vim
auto:
	cd tex && latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out -pvc main.tex

# Build the latex document
build:
	cd tex && latexmk -pdf -silent -pdflatex="pdflatex -interaction=nonstopmode" -output-directory=out main.tex

# Clean the latex build files
clean:
	cd tex && latexmk -c -output-directory=out


# Job submit
job:
	kubectl create -f job.yml
job-stop:
	runai delete job testjob
job-restart: job-stop job
	sleep 30s

# Shells
shell:
	kubectl exec --stdin --tty testjob-0-0 -- /bin/zsh -c "tmux -u attach || tmux -u"
zsh:
	kubectl exec --stdin --tty testjob-0-0 -- /bin/zsh

# Rsync

send-files:
	cd image-hijacks && git ls-files | $(KRSYNC) -azP --stats --files-from=- . testjob-0-0:/scratch/diego/image-hijacks

backup:
	$(KRSYNC) -azP --stats --exclude-from=rsync-exclude.txt testjob-0-0:/scratch/diego/semester-proj ./code

.PHONY: figures auto build clean job job-stop job-restart shell zsh send-files backup
