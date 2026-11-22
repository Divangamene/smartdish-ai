# SmartDish AI

Projeto de Deep Learning com Redes Neuronais Convolucionais (CNN).

## Estrutura

```
smartdish-ai/
├── configs/              # Configurações YAML (hiperparâmetros, caminhos)
├── data/
│   ├── raw/              # Dados originais, nunca alterar
│   ├── interim/          # Dados em transformação
│   ├── processed/        # Dados prontos: train/ val/ test/ -> <classe>/*.jpg
│   └── external/         # Datasets de terceiros
├── models/checkpoints/   # Pesos treinados (.pt, .keras)
├── notebooks/            # Jupyter notebooks de exploração
├── outputs/
│   ├── figures/          # Gráficos
│   ├── logs/             # TensorBoard
│   └── predictions/      # Resultados de inferência
├── scripts/              # Scripts utilitários
├── src/smartdish/        # Código-fonte do pacote
│   ├── data/             # Datasets, transforms, dataloaders
│   ├── models/           # Arquiteturas CNN
│   ├── training/         # Loops de treino e avaliação
│   └── utils/            # Funções auxiliares
└── tests/                # Testes (pytest)
```

## Ambiente

Gerido com [uv](https://docs.astral.sh/uv/) e Python 3.12 (PyTorch em CPU + TensorFlow/Keras).

```bash
uv sync                                  # instalar / atualizar dependências
uv run python scripts/check_env.py       # verificar o ambiente
uv run jupyter lab                       # abrir Jupyter
uv run pytest                            # correr testes
uv run python -m smartdish.training.train --config configs/default.yaml
uv run tensorboard --logdir outputs/logs
uv add <pacote>                          # adicionar nova biblioteca
```

No VS Code, seleciona o interpretador `.venv\Scripts\python.exe`.
