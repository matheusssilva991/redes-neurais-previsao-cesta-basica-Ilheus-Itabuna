# Configurar ambiente 
## Criar Ambiente
- conda create -n cesta python=3.8.10  
## Instalar Bibliotecas
- conda install -n cesta jupyterlab=3.0.14 (Opcional)
- conda install -n cesta tensorflow=2.7.0	
- conda install -n cesta pandas=1.2.4
- conda install -n cesta openpyxl
- conda install -n cesta -c conda-forge -c pytorch u8darts-all   #apenas se utilizar os modelos do Framework Darts
- conda install -n cesta matplotlib
## Ativar Ambiente
- conda activate cesta
## Desativar ambiente
- conda deactivate cesta

# Quick Start		
## Entrar no jupyterlab
- Digite no cmd: jupyter lab
## Arquivos de Configuração
- configuracoes.conf: Mudar cidade/Diretório de trabalho/Quantidade de Meses a serem previstos
## Executar Notebooks modelos
- Executar notebooks model_{nome do modelo}
## Resultados/Graficos
- configuracoes_grafico.config: Selecionar modelo_atual, mudar meses anteriores e meses a serem previstos
- Executar notebook grafico_cesta e grafico_produtos