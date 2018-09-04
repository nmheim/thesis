TARGETS = main.pdf
SOURCES = $(shell find . -name '*.tex')

main.pdf: main.tex $(SOURCES)
	latexmk -pdf -use-make main.tex

clean:
	latexmk -CA
	rm main.bbl
	rm -r _minted-main
	rm main.run.xml

open:
	evince $(TARGETS) &
