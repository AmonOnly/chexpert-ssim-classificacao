# Makefile para compilação do relatório LaTeX

.PHONY: all clean graficos figuras relatorio simples view help

# Variáveis
TEX_FILE = relatorio.tex
TEX_SIMPLES = relatorio_simples.tex
PDF_FILE = relatorio.pdf
PDF_SIMPLES = relatorio_simples.pdf
GRAFICOS_SCRIPT = gerar_graficos.py
FIGURAS_SCRIPT = gerar_figuras.py

# Compilador LaTeX (pdflatex ou xelatex)
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Python
PYTHON = python3

# Regra padrão
all: graficos relatorio

# Gera gráficos
graficos:
	@echo "==================================="
	@echo "Gerando gráficos..."
	@echo "==================================="
	$(PYTHON) $(GRAFICOS_SCRIPT)
	@echo ""

# Gera figuras de exemplo
figuras:
	@echo "==================================="
	@echo "Gerando figuras de exemplo..."
	@echo "==================================="
	$(PYTHON) $(FIGURAS_SCRIPT)
	@echo ""

# Compila o relatório (executa 2x para referências)
relatorio: $(TEX_FILE)
	@echo "==================================="
	@echo "Compilando relatório LaTeX..."
	@echo "==================================="
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)
	@echo "\nSegunda passagem (referências)..."
	$(LATEX) $(LATEX_FLAGS) $(TEX_FILE)
	@echo "\n✅ Relatório compilado: $(PDF_FILE)"

# Compila o relatório simples
simples: figuras $(TEX_SIMPLES)
	@echo "==================================="
	@echo "Compilando relatório simples..."
	@echo "==================================="
	$(LATEX) $(LATEX_FLAGS) $(TEX_SIMPLES)
	@echo "\nSegunda passagem (referências)..."
	$(LATEX) $(LATEX_FLAGS) $(TEX_SIMPLES)
	@echo "\n✅ Relatório simples compilado: $(PDF_SIMPLES)"

# Limpa arquivos auxiliares
clean:
	@echo "Limpando arquivos auxiliares..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
	@echo "✅ Limpeza concluída"

# Remove tudo (incluindo PDF e gráficos)
distclean: clean
	@echo "Removendo PDF e gráficos..."
	rm -f $(PDF_FILE)
	rm -rf figuras/
	@echo "✅ Limpeza completa"

# Abre o PDF (Linux)
view: relatorio
	@echo "Abrindo PDF..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF_FILE); \
	elif command -v evince > /dev/null; then \
		evince $(PDF_FILE) &; \
	elif command -v okular > /dev/null; then \
		okular $(PDF_FILE) &; \
	else \
		echo "⚠️  Nenhum visualizador de PDF encontrado"; \
	fi

# Ajuda
help:
	@echo "Makefile para Relatório LaTeX"
	@echo ""
	@echo "Uso: make [target]"
	@echo ""
	@echo "Targets disponíveis:"
	@echo "  all        - Gera gráficos e compila relatório técnico (padrão)"
	@echo "  graficos   - Gera apenas os gráficos em PDF"
	@echo "  figuras    - Gera apenas as figuras de exemplo"
	@echo "  relatorio  - Compila apenas o relatório técnico"
	@echo "  simples    - Gera figuras e compila relatório simples"
	@echo "  view       - Compila e abre o PDF técnico"
	@echo "  clean      - Remove arquivos auxiliares (.aux, .log, etc)"
	@echo "  distclean  - Remove tudo (PDF + gráficos + aux)"
	@echo "  help       - Mostra esta mensagem"
	@echo ""
	@echo "Exemplos:"
	@echo "  make              # Gera tudo (relatório técnico)"
	@echo "  make simples      # Gera relatório simples"
	@echo "  make view         # Compila e abre PDF técnico"
	@echo "  make clean        # Limpa arquivos temporários"
