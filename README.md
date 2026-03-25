# Balanceamento de Classificação para Imagem

Uma ferramenta inteligente e modular para balanceamento de datasets de classificação de imagem, projetada para preparar dados para treinamento de modelos de deep learning de forma robusta e automatizada.

## 🚀 Funcionalidades

- **Undersampling (Mode: `drop`)**: Reduz classes majoritárias para igualar o tamanho da menor classe.
- **Oversampling (Mode: `augment`)**: Aumenta classes minoritárias através de técnicas de data augmentation para igualar a maior classe.
- **Balanceamento Híbrido (Mode: `hybrid`)**: Equilibra o dataset utilizando um alvo intermediário, minimizando perda de dados e excesso de síntese.
- **Relatórios Detalhados**: Gera um log JSON completo com a distribuição original e pós-balanceamento.
- **Segurança e Reprodutibilidade**: Suporte a `dry_run`, `random_seed` e opção de cópia ou movimentação de arquivos.

## 📦 Instalação

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 🛠️ Exemplo de Uso

```python
import json
from balancer import FolderBalanceConfig, FolderImageDatasetBalancer

config = FolderBalanceConfig(
    input_root="/caminho/para/dataset",
    output_root="/caminho/para/dataset_balanceado",
    mode="hybrid",
    final_tolerance=0.10,
    intermediate_tolerance=0.30,
    random_seed=42,
    copy_instead_of_move=True,
    dry_run=False,
)

balancer = FolderImageDatasetBalancer(config)
report = balancer.run()

print(json.dumps(report, indent=2, ensure_ascii=False))
```

## ⚖️ Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
