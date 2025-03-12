1)	Refinar a simulação e Revisão Literária - Março
2)	Formular uma política baseada em planejamento - Abril
3)	Escrever TG 1 – Maio
4)	Escrever TG 2 – Junho
5)	Revisão Literária - Agosto
6)	Formular uma política baseada em RL - Setembro
7)	Escrever Dissertação – Outubro e Novembro


# TODO

ajustar pos behavior clone

Agressividade deve ser 1 quando muito próximo da área de interesse

Incluir o botão de regress

Escrever CADN

-------------------------------------------------------------
RL

Comando e controle: Eleição de lider para controle de solo

Implementar anti-colisão (para responder pergunta de curioso)

Escrever README
-------------------------------------------------------------

1)	Revisão Literária, Formular uma política baseada em RL e artigo Minor - Março
2)	Escrever TG 1 – Abril
3)	Escrever TG 2 – Maio
4)  Escrever Minor - Junho
5)	Revisão Literária - Agosto
7)	Escrever Dissertação – Setembro e Outubro


--------------------------------------

Estrutura de apresentação CADN
--------------------------------------
Uso emergente de drones de baixo custo como arma em conflitos armados recentes (Ucrânia vs Rússia)
Guerra como um jogo de estratégico, logistico e de otimização

Propor o simulador


Cenário Exemplo



--------------------------------------

Estrutura de apresentação do TG
--------------------------------------

Uso emergente de drones de baixo custo como arma em conflitos armados recentes (Ucrânia vs Rússia)
Guerra como um jogo de estratégico, logistico e de otimização

Propor uma uma possivel solução que faça frente a soluções centralizadas: uso de drones de baixo custo, autonomo com carater defensivo


Dentro os inumeros requisitos, que o desafio interdisciplinar de conceber uma plataforma dessas, um deles é o algortimo de coordenação, navegação e perseguição e engajamento
(aqui delimitar escopo)

Estudos/Presquisas relacionadas

Metodologia: Propor simulador, estratégias (5! 1º: mina aérea, 2º: descentralizado e sem comunicação, 3º: detec. descentralizado com comunicação, 4º: 3º: detec. descentralizado com comunicação + AEW, 5º: detec. centralizada com comunicação) e metrica de avaliação.

-> Simulador

-> Estratégia

-> Metrica de avaliação

Resultados

Conclusão


--------------------------------------

Estrutura de apresentação do MESTRADO
--------------------------------------

Uso emergente de drones de baixo custo como arma em conflitos armados recentes (Ucrânia vs Rússia)
Guerra como um jogo de estratégico, logistico e de otimização

Propor uma uma possivel solução que faça frente a soluções centralizadas: uso de drones de baixo custo, autonomo com carater defensivo


Dentro os inumeros requisitos, que o desafio interdisciplinar de conceber uma plataforma dessas, um deles é o algortimo de coordenação, navegação e perseguição e engajamento
(aqui delimitar escopo)

Estudos/Presquisas relacionadas

Metodologia: Propor simulador, estratégias (5! 1º: mina aérea, 2º: descentralizado e sem comunicação, 3º: detec. descentralizado com comunicação, 4º: 3º: detec. descentralizado com comunicação + AEW, 5º: detec. centralizada com comunicação) e metrica de avaliação.

-> Simulador

-> Estratégia

-> Metrica de avaliação

Resultados

Conclusão


--------------------------------------
Simulador
--------------------------------------

Simulador 2D para drones autônomos

Elementos:
Tela de simulação: Area de simulação e area de visualização de estados dos drones
na area de simulação é possivel ver a evolução da simulação e na area de visualização de estados dos drones é possivel ver o estado de cada drone. Ao clicar em algum drone desfensivo na area de simulação, é possivel trocar o drone que alimenta de informaçãoes a area de visualização.

Área de simulação: Na area de simulação, são representados graficamente drones atacantes, defensivos, uma áera de interesse,  elementos de solo (comando e controle e radar), metatadados da simulação (orientação de eixos, data e hora, duração, métricas de avaliação, etc)

Área de vizualização de estados de drones: Na area de vizualização de estados de drones, são representados graficamente as informações de estado parcial, que é tudo que o drone defensivo tem de conhecimento do ambiente. Essas informações podem ser obtidas através de detecção por meios proprios de drones amigos e inimigos e localização própria, ou pode ser obtida por meio de comunicação com outros drones defensivos que estejam em seu raio de comunicação. São esses os elementos que compõem a área de vizualização de estados: Matriz de recência de detecção, matrix de direção de detecção e posição do próprio drone e botões de controle de simulação.

Informações de estado parcial:
A proposta do do simulador é permitir que sejam possíveis, com as devidas simplificações, simular um combate entre times (enxame) de drones os drones defensivos não dispõem de informações completas do ambiente, mas sim de informações parciais. Será discutido então quais e como são essas informações.

Matriz de recência: A área de simulação é um retangulo onde cada ponto pode ser representado por um vetor no R² de números reais (para fins práticos, disconsidere a discretização que é feita pela representação binária no computador). A matriz de recência é criada a partir da discretização da área de simulação em uma grade de células de mesma dimensão. Cada célula da matriz de recência é um número real entre 0 e 1 que representa a recência da detecção de um drone inimigo ou amigo. A recência é um valor que varia de 0 a 1, onde quando mais próximo de 0 significa que a detecção é muito antiga e de 1 significa que a detecção é muito recente. A matriz é atualizada a cada passo de simulação, de acordo com um fator de decaimento exponencial para as detecções já feitas e eventualmente, com as novas detecções que são realizadas.

Matriz de direção: A matriz de direção é criada a partir da discretização da área de simulação em uma grade de células de mesma dimensão. Cada célula da matriz de direção é um vetor no R² de números reais que representa a direção da detecção de um drone inimigo ou amigo. A direção é um vetor que aponta para a direção do drone detectado. A matriz é atualizada a cada passo de simulação, de acordo com a direção do drone detectado.

Ambas as matrizes possuem uma matriz auxiliar de mesma dimensão que armazena o momento da atualização de cada célula. Elas não constituem por si só informação extra na representação do estado, pois tanto a matriz de recencia quanto a matriz de direção somente retem a informação mais atual de cada célula (isso é, o que ele acha da célula. Não se conhece o real estado de detecção de uma célula que esteja além do raio de detecção). Informação de carater temporal é obtido da interpretação direta da matriz de recencia (0 a 1).

Posição do próprio drone: A posição do próprio drone é um vetor no R² de números reais que representa a posição do drone defensivo na área de simulação. A posição é atualizada a cada passo de simulação, de acordo com a direção do drone detectado.

Área de vizualização de estados de drones: Na area de vizualização de estados de drones, são representados graficamente as informações de estado parcial. As informações de recencia e direção são condensadas numa única represetanção gráfica (plotagem de superfície 3D), onde a altura da superficie representa a recência e a direção é representada pela cor. A posição do próprio drone é representada por uma linha vertical preta que atravessa a superficie. Naturalmente, sem o devido tratamento, essa superficie se assemelha a um campo com abruptas elevações pontuais. Por tanto, para que a visualização seja mais clara, é feita uma suavização da superfície por meio de uma convolução com uma superficie gaussiana de parâmetros apropriados. Dessa forma, o estado interno do drone é representado de forma clara e intuitiva em dois gráficos diferentes, um destinado para drones do mesmo time/enxame (defensivo) e outro para os drones inimigos (atacantes).

Dinâmica da troca de informações: A troca de informações entre drones defensivos é feita por meio de comunicação ad hoc. A comunicação ad hoc é uma rede de comunicação sem fio que não necessida de uma infraestrutura centralizada para funcionar. Os componentes mínimos, necessários e suficientes são os nós da rede. A comunicação ad hoc é feita por meio de um raio de comunicação, que é a distância máxima que um drone defensivo pode se comunicar com outro drone defensivo. A comunicação é feita por meio de mensagens que contém as matrizes de recência e direção do drone que envia a mensagem. Ao receber um mensagem de um drone amigo, o drone receptor atualiza suas matrizes de recência e direção com as informações recebidas. Essa atualização é feita célula a céluna, de modo que seja mantido a informação mais atual disponível de cada célula (ou a própria ou a do amigo). Além do filtro de qual informação é mais atual, também é feito correções do tipo: O drone que recebe a mensagem avalia se o que é informado pelo drone amigo com respeito a sua área de detecção (drone que recebe) é coerente, já que nesse momento, ele tem a "verdade" inquestionavel daquela porção de ambiente, armazendo essa atualização com datetime de atualização mais recente. Perceba que, por ser um algoritmo distribuido, ora o drone que recebe a mensagem é o drone que envia a mensagem, ora o drone que envia a mensagem é o drone que recebe a mensagem. E efeito resultante disso é que todos os drones que juntos formam uma rede possuem a mesma informação de estado parcial do ambiente com tempo de atualização exato de um passo de simulação.

Elementos de Simulação

Área de Interesse:
A área de interese é a região alvo de ataque por parte dos drones inimigos e que devem ser defendidos pelos drones defensivos. A área de interesse é representada por um círculo de raio interno e externo. O raio interno é uma referência visual de zona crítica a ser mantida livre de inimigos a todo custo. O raio externo é o limite de atuação dos drones amigos, até onde eles devem se manter para defender o ponto de interesse. A cor do círculo varia dinamicamente de verde a vermelho conforme a saúde do ponto de interesse diminui, refletindo os danos acumulados. A saúde do ponto de interesse é um valor numérico de 0 a 100. O dano referente a cada ataque bem sucedido será detalhado na subseção relativa ao Drone inimigo e seu comportamento.

Drone inimigo:
Os drones inimigos são drones que atacam o ponto de interesse. Eles partem de ponto aleatórios das bordas da área de simulação e se movem em direção a Área de Interesse. Essa trajetória pode ser de várias maneiras diferentes e são modeladas pra aumentar o grau de imprevisibilidade do time defensivo. No momento da instanciação desses drones, além da posição inicial, seus comportamentos de trajetória são definidos aleatoriamente. Os Drones inimigos não dispõem de comunicação tal como os drones inimigos pois são concebidos para simulador o seu uso típico em cenários de ataque que é sendo remotamente controlado por um grupo de humanos e estes sim trocam informação. Os drones inimigos possuem mecanismo de detecção de drones amigos de curto alcançe, simulando meios de detecção baseado em visão computacional, haja visto a baixo rastreabilidade por assinatura radar desse vetor. O drone inimigo então segue sua trajetória estocástica até a área de interesse, até que algum dos eventos abaixo acontecem:

- detecte algum drone defensivo: Nesse caso o drone inimigo avalia um valor aleatório de 0 a 1 (distribuição uniforme) comparado ao parâmetro de agressividade. Se o valor aleatório for menor que o parâmetro, o drone entra em modo ataque desesperado (desperate_attack), alterando sua direção de movimento para atacar diretamente a área de interesse em linha reta; caso contrário, ele ativa um modo de fuga, desviando-se na direção oposta ao drone defensivo por um número fixo de passos de simulação antes de retomar sua trajetória original. O parametro de agressividade é um valor dinâmico, se tornando maior conforme o drone se aproxima da área de interesse

- entra no raio de engajamento de um drone defensivo: Esse evento é disparado quando um drone atacante e defensivo entram no raio de engajamento um do outro. Nesse caso, existem 3 possiveis desfechos:

a) O drone defensivo neutraliza o drone inimigo sem sofrer danos e segue na sua tarefa de defesa

b) O drone inimigo neutraliza o drone defensivo, removendo ambos da simulação

c) O drone defensivo sofre falha interna ou escasces de meios e é removido, enquanto o drone inimigo continua sua aproximação ofensiva.

- Por fim, quando o drone inimigo adentra o raio de engajamento do centro da área de interesse. Nesse caso, o drone inimigo ataca a área de interesse e a saúde do ponto de interesse é reduzida. O dano que cada drone inimigo pode causar à área de interesse é 100/numero de drones inimigos.

Drone defensivo

Drones defensivos são drones responsaveis por defender a área de interesse. Inicialmente são instanciados ao redor da área de interesse, frequentemente distribuídos em forma de polígono regular para cobrir uma área estratégica. Os drones defensivos possuem mecanismo de detecção de drones inimigos de curto alcançe, simulando meios de detecção baseado em visão computacional, haja visto a baixo rastreabilidade por assinatura radar desse vetor.

Os drones amigos possuem um comportamento programável definido por politica (policy) que, dado o seus estado (sua representação parcial do ambiente), determina a ação a ser tomada. A politica é uma função que mapeia o estado do drone para uma ação. A politica pode ser implementada por meio de um algoritmo de planejamento, onde descreve-se a ação a ser tomada para cada estado possível; por aprendizado de máquina supervisionado (behavior clone) ou por aprendizado por reforço, onde o exame de drones aprende com a experiencia adquida com inúmeras interação com o ambiente. A ação é um vetor no R² de números reais que representa a direção do movimento do drone. A ação é atualizada a cada passo de simulação, de acordo com a politica do drone.

Drone defensivo de alerta antecipado (AEW):
São drones defensivos que possuem um raio de detecção maior que os drones defensivos comuns. Eles são instanciados ao redor da área de interesse, frequentemente distribuídos em forma de polígono regular para cobrir uma área estratégica. Sua movimentação e puramente de orbitar a área de interesse, sem nunca perseguir inimigos ou então desviar-se da sua rota circular. Desse forme, eles conseguem prover a rede de comunicação distribuida de informações de detecção para regiões que extrapolam o raio exterior da área de interesse. Eles não possuem mecanismo de ataque e quando entram mutualmente na area de engajamento de um drone inimigo reage da seguinte forma:
amostra-se um numero de 0 a 1 advindo de uma distribuição uniforma. se esse valor for inferior a uma constante de eliminação mutual, ambos os drones são removidos da simulação. Caso contrário, nada acontece e os drones continuam em suas trajetórias específicas.
 
Radar de detecção de solo (RADAR)
O radar de detecção de solo é um elemento de solo que detecta drones inimigos e amigos em sua área de detecção, tipicamente bem abrangente e retransmite na rede de comunicação ad hoc. Por ser um elemento valioso ao correto funcionamento da estratégia de defesa, ele normalmente é instanciado dentro da área de interesse.


Abstrações de Simulações
alem do que foi descrito acima, a simulação possui abstrações que tornam possivel a implementação e a execução da simulação. São elas:
ambiente 2D, por entender que o proposito do drone é puramente defensivo (detecção, perseguição e neutralização de drones igualmente de pequeno porte), que difere com a forma que drones atacantes vem sendo usado (com carga significativa de explosivos), assume-se que os drones defensivos conseguem se locomover com velocidade 10 % superior aos drones atacantes.