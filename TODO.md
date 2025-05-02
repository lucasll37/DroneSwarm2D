3)	Escrever TG 1 – Maio
4)	Escrever TG 2 – Junho
5)	Revisão Literária - Agosto
6)	Formular uma política baseada em RL - Setembro
7)	Escrever Dissertação – Outubro e Novembro


# TODO

- Cenários:
1) greedy
2) centralizada
3) proposta
4) proposta com aew


- friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates] # if 1.1 * INITIAL_DISTANCE - candidate_pos.distance_to(INTEREST_POINT_CENTER) > 0] ????

- implementar TRIANGULAÇÃO inplace
- Implementar anti-colisão (para responder pergunta de curioso)
-------------------------------------------------------------

Behaviorclone do spread e RL
- Velocidade Variável ()
-------------------------------------------------------------


Reunião com o professor

- Apresentar resultados dos cenários
- Pedir para o trabalho valer para o PFC-Ciência de Dados
- Pedir uso de IA para corrigir erros ortográficos e documentar código fonte
- Mostrar paper submetido ao CADN
- Acordar entregável do TG1: Introdução, Revisão Literária, Metodologia e Apresentação do Simulador
- Acordar entregável do TG2: Resultados e Conclusão
- Propor banca (Alonso e Juliana) e data 
- Apresentar proposta de Mestrado:
    - Maior parte do tempo, Drones ficam no estado de espera (HOLD)
    - Fazer uso de RL para otimizar o comportamento nesse estado (prováveis comportamentos emergentes: ronda e distribuição ao redor da área de interesse)
    - Uso dos algortimos:
        - QMIX (Q-Mixing Network)
        - COMA (Counterfactual Multi-Agent Policy Gradients)
        - MAPPO (Multi-Agent Proximal Policy Optimization)
        - VDN (Value Decomposition Networks)
        - MAT (Multi-Agent Transformer)
    - Comparar com o algoritmo de planejamento com hold estático (TG)
    - Métrica de avaliação: Danos causados no alvo
    - Recompensa de aprendizagem: A mesma usada no TG (Composição: Danos causados no alvo, Distância média do alvo, Drones abatidos e Tempo de episódio)
    - Comportamento de espera benchmark:
        - Espera no ponto atual até nova designação de atuação
        - Movimento browniano (aletório) até nova designação de atuação

- Propor submissão de resultados do Mestrado antes da defesa no Jornal IEEE Latin America Transactions (Título e resumo em inglês, restante em português) ($250 dolares)

Cronograma:
1)	Escrever TG 1 – Maio
2)	Escrever TG 2 – Junho
3)	Revisão Literária RL - Agosto
4)	Formular políticas baseada em RL - Setembro
5)	Escrever Dissertação – Outubro e Novembro
6)	Submissão do artigo para o Jornal IEEE Latin America Transactions – Novembro
7)	Defesa do Mestrado – ASAP 2026


_____________________________________________________

Bom dia, Prof. Dr. Tasinaffo.

Conforme conversamos na última reunião, gostaria de acordar com o senhor quando ao uso de IA generativa para corrigir erros ortográficos, semânticos e de concordância verbal e nominal das produções literárias, bem como para fazer a geração de documentação do código fonte do projeto que envolve o Trabalho de Graduação (TG) e Dissertação (Mestrado).

Respeitosamente,

Lucas Lima
_____________________________________________________
Bom dia, Prof. Dr. Paulo André.

Gostaria de atualizar a minha intenção de cumprimir as atividades previstas para a obtenção da Formação Complementar em Ciência de Dados.

Previamente, em 2023, manifestei a intenção para o então coordenador do PDC-Dados, bem como apresentei meu projeto de pesquisa. Contudo, gostaria de fazer modificações de modo a aproveitar o trabalho que venho desenvolvendo junto ao professor Tasinaffo, no PMG. Brevemente, o Trabalho possui a seguinte proposta:


A intersecção com a área de ciência de dados se dá no uso do tratamento de dados do que for obtido das simulações, testes estatisticos de validação e comparação e o desenvolvimento do ambiente de simulação que é compatível com a biblioteca OpenAI Gym, o que possibilita o uso de algoritmos de aprendizado por reforço (RL) para otimização do comportamento dos drones.

O Professor nós lê em cópia e foi de acordo com a proposta.

Desde já, agradeço a atenção e fico à disposição para qualquer dúvida.

Respeitosamente,

Lucas Lima

_____________________________________________________

Bom dia Prof. Dr. Tasinaffo.

Segue em anexo o artigo que submeti ao CADN. Na sequência segue as informações que tratamos na nossa última reunião e o senhor pediu para eu enviar por email.:

- Proposta de Mestrado:
    - Maior parte do tempo, Drones ficam no estado de espera (HOLD)
    - Fazer uso de RL para otimizar o comportamento nesse estado (prováveis comportamentos emergentes: ronda e distribuição ao redor da área de interesse)
    - Uso de algum dos algortimos:
        - QMIX (Q-Mixing Network)
        - COMA (Counterfactual Multi-Agent Policy Gradients)
        - MAPPO (Multi-Agent Proximal Policy Optimization)
        - VDN (Value Decomposition Networks)
        - MAT (Multi-Agent Transformer)

    - Comparar com o algoritmo de planejamento com hold estático (Resultado do TG)
    - Métrica de avaliação: Danos causados no alvo
    - Recompensa de aprendizagem: A mesma usada no TG (Composição: Danos causados no alvo, Distância média do alvo até à área de interesse, Número de Drones abatidos e Tempo de        episódio)
    - Comportamento de espera benchmark:
        - Resultado do TG
        - Movimento browniano (aletório) até nova designação de atuação


Cronograma:
1)	Escrever TG 1 – Maio
2)	Escrever TG 2 – Junho
3)	Revisão Literária RL - Agosto
4)	Formular políticas baseada em RL - Setembro
5)	Escrever Dissertação – Outubro e Novembro
6)	Submissão do artigo para o Jornal IEEE Latin America Transactions – Novembro
7)	Defesa do Mestrado – ASAP 2026