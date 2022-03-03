Criar ambiente (Importante, caso contrario pode causar conflito de dependencias)
	conda create -n cesta python=3.8.10  

Instalar Bibliotecas
	conda install -n cesta jupyterlab=3.0.14
	conda install -n cesta tensorflow=2.7.0	
	conda install -n cesta pandas=1.2.4
	conda install -n cesta openpyxl
	conda install -n cesta -c conda-forge -c pytorch u8darts-all
	conda install -n cesta matplotlib

Ativar Ambiente
	conda activate cesta
Entrar no jupyterlab
	jupyter lab

Quick Start
	configuracoes.conf: Mudar cidade/Diretório de trabalho/Quantidade de Meses a serem previstos

	Executar notebooks model_{nome do modelo}

	Pasta resultados
		Executar notebook resultados_previsoes