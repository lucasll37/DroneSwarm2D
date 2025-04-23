# DroneSwarm2D: Um Simulador para Estudo de Táticas Defensivas Distribuídas com Enxames de Drones Autônomos

![Banner do Enxame de Drones](./images/banner.png)

## 📚 Visão Geral

DroneSwarm2D é um ambiente de simulação 2D projetado para estudar táticas defensivas distribuídas para enxames de drones autônomos. O simulador aborda o crescente desafio representado por drones ofensivos de baixo custo em conflitos modernos, possibilitando a pesquisa de estratégias de defesa descentralizadas e resilientes. Ele fornece uma plataforma para modelagem de comportamentos de enxames de drones, implementação de redes de comunicação ad-hoc e avaliação da eficácia de vários algoritmos defensivos.

### Características Principais

- **Tomada de Decisão Distribuída**: Modela drones autônomos que operam sem controle central
- **Comunicação em Rede Ad-hoc**: Simula comunicações realistas entre drones
- **Percepção de Estado**: Cada drone mantém sua própria percepção do ambiente através de detecção local
- **Compartilhamento de Informações**: Os drones trocam e mesclam matrizes de detecção para melhorar a consciência colaborativa
- **Múltiplos Comportamentos**: Implementa vários padrões de aproximação inimiga e estratégias de resposta defensiva
- **Visualizações**: Ferramentas ricas de visualização para matrizes de detecção, estados dos drones e estatísticas
- **Métricas de Desempenho**: Métricas abrangentes para avaliar a eficácia da defesa

## 🎯 Motivação do Projeto

A crescente acessibilidade de drones de baixo custo transformou os cenários de conflito modernos. Esses dispositivos de baixo custo, frequentemente construídos com materiais simples e adaptados para fins ofensivos, representam desafios significativos para os sistemas de defesa convencionais, que normalmente exigem investimentos substanciais e frequentemente têm dificuldades para enfrentar ataques em enxame.

O DroneSwarm2D aborda essa assimetria explorando redes descentralizadas de defesa de drones que:

1. Eliminam pontos únicos de falha (comuns em sistemas centralizados)
2. Fornecem alternativas econômicas às contramedidas tradicionais caras
3. Permitem respostas flexíveis e escaláveis a diversas ameaças
4. Otimizam a coordenação tática através de princípios de computação distribuída

Esta abordagem se inspira em princípios operacionais de pesquisa de longa data, visando criar sistemas de defesa resilientes que possam operar efetivamente mesmo com recursos limitados.

## 📂 Organização do Projeto

```
DroneSwarm2D/
│
├── assets/                      # Recursos visuais e de interface
│   ├── base_*.svg               # Imagens SVG para base
│   ├── drone_*.svg              # Imagens SVG para drones
│   ├── radar_*.svg              # Imagens SVG para radares
│   └── markdown.css             # Estilo para documentos markdown
│
├── config/                      # Configurações da simulação
│   ├── preset/                  # Configurações predefinidas
│   │   ├── airmine.json         # Configuração para cenário padrão
│   │   ├── proposal.json        # Configuração para a solução proposta
│   │   ├── centralized.json     # Configuração para abordagem centralizada
│   │   └── ...                  # Outras configurações
│   └── proposal_spread.json     # Configuração alternativa
│
├── data/                        # Dados e resultados das simulações
│   ├── airmine/                 # Resultados para cenário padrão
│   ├── proposal/                # Resultados para solução proposta
│   └── ...                      # Outros resultados
│
├── images/                      # Imagens para documentação
│
├── models/                      # Modelos de IA treinados
│
├── src/                         # Código-fonte
│   ├── environment/             # Núcleo do ambiente de simulação
│   │   ├── AirTrafficEnv.py     # Ambiente principal
│   │   ├── DemilitarizedZone.py # Zonas desmilitarizadas
│   │   ├── EnemyDrone.py        # Implementação de drones inimigos
│   │   ├── FriendDrone.py       # Implementação de drones amigos
│   │   ├── InterestPoint.py     # Ponto de interesse a ser defendido
│   │   ├── distributedDefensiveAlgorithm.py  # Algoritmo de defesa
│   │   ├── settings.py          # Configurações do ambiente
│   │   └── utils.py             # Funções utilitárias
│   │
│   ├── analysis.py              # Script para análise de resultados
│   ├── behavior_clone.py        # Script para clonagem de comportamento
│   └── main.py                  # Ponto de entrada principal
│
├── utils/                       # Utilitários adicionais
│   ├── create_video.py          # Cria vídeos das simulações
│   ├── view_enemy_behavior.py   # Visualiza comportamentos inimigos
│   └── ...                      # Outros utilitários
│
├── LICENSE                      # Licença do projeto
├── Makefile                     # Automação de tarefas
├── README.md                    # Este arquivo
└── requirements.txt             # Dependências do projeto
```

## 🔧 Arquitetura Técnica

### Ambiente de Simulação

A simulação é construída em um sistema de grade 2D com os seguintes componentes:

- **Ponto de Interesse**: Área central a ser defendida, com saúde que diminui quando atacada com sucesso
- **Drones Amigos**: Drones defensivos autônomos implementando vários comportamentos
- **Drones Inimigos**: Drones ofensivos com padrões de ataque configuráveis
- **Zonas Desmilitarizadas**: Áreas onde o engajamento é proibido

### Representação de Estado

Cada drone mantém sua própria percepção local do ambiente através de:

1. **Matrizes de Detecção**:
   - `enemy_intensity`: Registra detecções recentes de inimigos (valores 0-1)
   - `enemy_direction`: Armazena vetores de direção de inimigos detectados
   - `friend_intensity`: Registra detecções recentes de drones amigos
   - `friend_direction`: Armazena vetores de direção de drones amigos detectados

2. **Sistema de Triangulação**:
   Quando múltiplos drones detectam o mesmo alvo de diferentes ângulos, suas informações de detecção são combinadas para melhorar a precisão.

### Sistema de Comunicação

A simulação implementa comunicação realista entre drones com:
- Alcance de comunicação limitado entre drones
- Formação de rede ad-hoc baseada em proximidade
- Perda probabilística de mensagens
- Fusão de informações através da mesclagem de matrizes

### Comportamentos dos Drones

#### Comportamentos de Drones Defensivos
- **Planning**: Tomada de decisão estratégica baseada na percepção atual do estado
- **AEW (Alerta Aéreo Antecipado)**: Padrão de vigilância em órbita
- **RADAR**: Unidade de detecção estacionária
- **AI**: Tomada de decisão baseada em rede neural

#### Comportamentos de Drones Ofensivos
- **Direct**: Abordagem direta ao alvo
- **Zigzag**: Abordagem com oscilações laterais
- **Spiral**: Movimento em espiral em direção ao alvo
- **Alternating**: Alternância entre abordagem direta e movimento perpendicular
- **Bounce Approach**: Avançar e recuar com variação direcional
- **Formation**: Movimento coordenado em formações predefinidas

## 🚀 Como Iniciar

### Pré-requisitos

- Python 3.9+
- Bibliotecas: pygame, numpy, scipy, matplotlib, tensorflow, pandas

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seunome/DroneSwarm2D.git
cd DroneSwarm2D

# Crie um ambiente conda
make create_env

# Ative o ambiente conda
conda activate ./.venv

# Instale dependências do sistema (Ubuntu/Debian)
make apt_deps

# Ou instale diretamente com pip
pip install -r requirements.txt
```

### Executando a Simulação

```bash
# Execute a simulação principal
make run

# Ou diretamente com Python
python src/main.py

# Execute análise nos resultados
make analysis

# Criar vídeo a partir de frames salvos
make video

# Visualizar padrões de comportamento inimigo
make traj

# Visualizar comportamento de múltiplos inimigos
make multtraj
```

### Configuração da Simulação

O simulador utiliza arquivos JSON na pasta `config/preset/` para definir parâmetros de simulação. Você pode:

1. Modificar os arquivos existentes para ajustar parâmetros
2. Criar novos arquivos de configuração baseados nos existentes
3. Especificar um arquivo de configuração personalizado através da variável de ambiente `CONFIG_FILE`

```bash
# Executar com uma configuração específica
CONFIG_FILE=./config/preset/proposal_spread.json python src/main.py
```

## 🎮 Interface do Usuário

A interface da simulação consiste em:

1. **Área de Simulação**: Representação 2D do ambiente mostrando:
   - Drones amigos (brancos)
   - Drones inimigos (vermelhos)
   - Ponto de interesse (círculo verde que se torna vermelho conforme a saúde diminui)
   - Alcances de detecção (círculos tracejados)
   - Links de comunicação (linhas tracejadas)

2. **Painel de Visualização**: Visualização 3D da percepção de um drone selecionado:
   - Intensidade e direção de detecção de inimigos
   - Intensidade e direção de detecção de amigos
   - Ângulos de direção codificados por cores

3. **Painel de Controle**: Botões de interface para:
   - Alternar recursos de visualização
   - Pausar/retomar simulação
   - Exportar dados
   - Reiniciar a simulação

4. **Exibição de Estatísticas**: Métricas em tempo real mostrando:
   - Contagens de drones
   - Estatísticas de comunicação
   - Saúde do ponto de interesse
   - Tempo de simulação

## 📊 Descobertas da Pesquisa

O simulador permite a comparação entre diferentes estratégias defensivas:

1. **Benchmark (Não-cooperativo)**: Drones operam independentemente sem comunicação
2. **Proposta (Distribuída)**: Drones compartilham informações através de redes ad-hoc
3. **Centralizada**: Drones têm informação global completa

A análise dessas abordagens demonstra que:

- Abordagens distribuídas oferecem vantagens significativas sobre as não-cooperativas
- A diferença de desempenho entre abordagens distribuídas e centralizadas é mínima em muitos cenários
- Estratégias distribuídas mostram maior resiliência a falhas de comunicação e adaptações inimigas

Testes estatísticos (Kolmogorov-Smirnov e Mann-Whitney) confirmam diferenças significativas nas métricas de desempenho entre as abordagens.

## 💡 Componentes Algorítmicos Principais

### Algoritmo de Planejamento do Drone Amigo

```python
def planning_policy(state, ...):
    # Extrai informações do estado
    pos = state['pos']
    friend_intensity = state['friend_intensity']
    enemy_intensity = state['enemy_intensity'] 
    enemy_direction = state['enemy_direction']
    
    # Identifica e prioriza alvos inimigos
    enemy_targets = []
    for cell, intensity in np.ndenumerate(enemy_intensity):
        if intensity > activation_threshold:
            target_pos = cell_to_position(cell)
            distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
            enemy_targets.append((cell, target_pos, distance_to_interest))
    
    # Ordena alvos por proximidade ao ponto de interesse
    enemy_targets.sort(key=lambda t: t[2])
    
    # Atribui drones para interceptar alvos
    # [Lógica de interceptação...]
    
    # Para drones não atribuídos, implementa padrão de espera
    # [Lógica de padrão de espera...]
```

### Cálculo de Interceptação

```python
def intercept_direction(chaser_pos, chaser_speed, target_pos, target_vel):
    # Calcula posição relativa
    r = target_pos - chaser_pos
    
    # Resolve equação quadrática para tempo de interceptação
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    b = 2 * r.dot(target_vel)
    c = r.dot(r)
    
    # [Lógica de solução...]
    
    # Retorna vetor de direção normalizado
    return direction.normalize()
```

### Mesclagem de Matrizes para Fusão de Informações

```python
def merge_enemy_matrix(self, neighbor):
    # Atualiza com base na comparação de timestamp
    update_mask = neighbor.enemy_timestamp > self.enemy_timestamp
    np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
    np.putmask(self.enemy_direction, 
               np.broadcast_to(update_mask[..., None], self.enemy_direction.shape),
               neighbor.enemy_direction)
    np.putmask(self.enemy_timestamp, update_mask, neighbor.enemy_timestamp)
```

## 📈 Direções Futuras

O simulador DroneSwarm2D abre várias direções de pesquisa promissoras:

1. **Integração de Aprendizado por Reforço**: Implementação de algoritmos de aprendizado para melhorar táticas defensivas com base na experiência
2. **Estudos de Escalabilidade**: Investigação de desempenho com enxames muito maiores (centenas ou milhares de drones)
3. **Enxames Heterogêneos**: Modelagem de enxames com diversas capacidades e papéis de drones
4. **Protocolos de Comunicação Avançados**: Teste de protocolos especializados para redes drone-a-drone
5. **Extensão 3D**: Expansão da simulação para três dimensões para cenários mais realistas

## 🤝 Contribuindo

Contribuições para o DroneSwarm2D são bem-vindas! Sinta-se à vontade para enviar pull requests ou abrir issues para discutir possíveis melhorias.

## 📄 Licença

Este projeto é lançado sob a MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📧 Contato

Para perguntas ou oportunidades de colaboração, entre em contato pelo e-mail [lucas.silva1037@gmail.com].