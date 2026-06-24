.PHONY: all install run pdf clean

all: run pdf

install: .venv/touch

.venv/touch: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	touch .venv/touch

run: install
	@echo "Executando diagrama da arquitetura..."
	.venv/bin/python src/arch.py
	@echo "Executando plotagem do KMeans..."
	.venv/bin/python src/gen_kmeans.py
	@echo "Executando pipeline principal..."
	.venv/bin/python src/pipeline.py

pdf:
	pdflatex artigo.tex
	pdflatex artigo.tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk
	rm -rf __pycache__/ src/__pycache__/ .venv/