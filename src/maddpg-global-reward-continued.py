def integrate_with_simulation():
    """
    Exemplo de como integrar o MADDPG com um sistema de simulação existente.
    Esta função mostra os pontos de integração necessários.
    """
    # 1. Definir as constantes que seriam importadas do seu código
    GRID_WIDTH = 50
    GRID_HEIGHT = 50
    CELL_SIZE = 1.0
    MAX_VELOCITY = 10.0
    TRIANGULATION_GRANULARITY = 5
    
    # 2. Inicializar o agente MADDPG
    num_drones = 10
    state_dim = 30  # Tamanho do vetor de estado por drone
    action_dim = 4  # [strength, radius, damping, exploration_bias]
    action_bounds = (0.0, 5.0)
    
    state_dims = [state_dim] * num_drones
    action_dims = [action_dim] * num_drones
    maddpg_agent = GlobalRewardMADDPG(num_drones, state_dims, action_dims, action_bounds)
    
    # Carregar um modelo pré-treinado (opcional)
    try:
        maddpg_agent.load_models("maddpg_global_models_final")
        print("Modelo pré-treinado carregado com sucesso!")
    except:
        print("Nenhum modelo pré-treinado encontrado. Iniciando com pesos aleatórios.")
    
    # 3. Executar a simulação
    is_training = True  # Defina como False para apenas usar o modelo sem treinar
    episode_count = 0
    max_episodes = 1000
    
    # Loop principal da simulação (este seria integrado ao seu loop existente)
    while episode_count < max_episodes:
        # Inicializar/reiniciar o ambiente para um novo episódio
        drones = initialize_drones(num_drones)  # Esta seria sua função de inicialização
        enemies = initialize_enemies()          # Esta seria sua função de inicialização
        grid_coverage = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        # Valores iniciais para recompensa global
        prev_global_coverage = calculate_global_coverage(grid_coverage)
        prev_global_detection = calculate_global_detection(drones)
        
        # Executar um episódio
        step = 0
        max_steps = 200
        while step < max_steps:
            # Aqui você integraria com o seu loop de simulação existente
            
            # 1. Atualizar vizinhos (sua lógica existente)
            update_drone_neighbors(drones)
            
            # 2. Extrair estados para MADDPG
            states = [extract_state_for_drone(drone, drones, enemies, grid_coverage) for drone in drones]
            
            # 3. Obter ações do MADDPG
            actions = maddpg_agent.get_actions(states, add_noise=is_training)
            
            # 4. Aplicar ações aos drones
            for i, drone in enumerate(drones):
                # Quando um drone está em estado HOLD_SPREAD, use MADDPG
                if drone.state == "HOLD_SPREAD":
                    apply_maddpg_actions(drone, actions[i])
                # Caso contrário, use a lógica existente
            
            # 5. Atualizar os drones (sua lógica existente)
            update_drones(drones, enemies)
            
            # 6. Atualizar cobertura da grade (sua lógica existente)
            update_grid_coverage(drones, grid_coverage)
            
            # 7. Calcular recompensa global
            current_global_coverage = calculate_global_coverage(grid_coverage)
            current_global_detection = calculate_global_detection(drones)
            
            global_reward = calculate_global_reward(
                drones,
                prev_global_coverage,
                current_global_coverage,
                prev_global_detection,
                current_global_detection
            )
            
            # 8. Extrair próximos estados
            next_states = [extract_state_for_drone(drone, drones, enemies, grid_coverage) for drone in drones]
            
            # 9. Armazenar experiência se estiver treinando
            if is_training:
                done = (step == max_steps - 1)
                maddpg_agent.store_transition(states, actions, global_reward, next_states, done)
                
                # 10. Treinar o agente
                maddpg_agent.train()
            
            # Atualizar valores para próxima iteração
            prev_global_coverage = current_global_coverage
            prev_global_detection = current_global_detection
            
            step += 1
        
        # Fim do episódio
        if is_training and (episode_count + 1) % 50 == 0:
            maddpg_agent.save_models(f"maddpg_global_models_ep{episode_count+1}")
            print(f"Modelo salvo após {episode_count+1} episódios")
        
        episode_count += 1
    
    # Salvar modelo final
    if is_training:
        maddpg_agent.save_models("maddpg_global_models_final")
        maddpg_agent.plot_results()


def calculate_global_coverage(grid_coverage):
    """
    Calcula a cobertura global da grade.
    
    Args:
        grid_coverage: Matriz de cobertura da grade
    
    Returns:
        Porcentagem de células cobertas
    """
    total_cells = grid_coverage.shape[0] * grid_coverage.shape[1]
    covered_cells = np.sum(grid_coverage > 0)
    return covered_cells / total_cells


def calculate_global_detection(drones):
    """
    Calcula a detecção global de inimigos.
    
    Args:
        drones: Lista de drones ativos
    
    Returns:
        Número total de células detectadas
    """
    # Combinando todas as matrizes de detecção passiva
    if not drones:
        return 0
    
    # Inicializa com a primeira matriz
    combined_matrix = drones[0].passive_detection_matrix.copy()
    
    # Combina com as demais matrizes usando OR lógico
    for drone in drones[1:]:
        combined_matrix = np.logical_or(combined_matrix, drone.passive_detection_matrix)
    
    # Retorna o número total de células detectadas
    return np.sum(combined_matrix)


# Implementação de um comportamento clonado para inicializar o MADDPG
def behavior_cloning_initialization(maddpg_agent, drones, enemies, grid_coverage, num_samples=1000):
    """
    Inicializa o buffer de replay com amostras do comportamento atual de repulsão.
    
    Args:
        maddpg_agent: Agente MADDPG a ser inicializado
        drones: Lista de drones para coletar experiência
        enemies: Lista de inimigos no ambiente
        grid_coverage: Matriz de cobertura da grade
        num_samples: Número de amostras a coletar
    """
    print("Inicializando com clonagem comportamental...")
    
    # Valores iniciais
    prev_global_coverage = calculate_global_coverage(grid_coverage)
    prev_global_detection = calculate_global_detection(drones)
    
    samples_collected = 0
    
    while samples_collected < num_samples:
        # Atualizar vizinhos
        update_drone_neighbors(drones)
        
        # Extrair estados
        states = [extract_state_for_drone(drone, drones, enemies, grid_coverage) for drone in drones]
        
        # Extrair ações atuais (da estratégia de repulsão existente)
        actions = []
        for drone in drones:
            # Capturar os parâmetros atuais de repulsão como "ações"
            action = [
                drone.repulsion_params['strength'],
                drone.repulsion_params['radius'],
                drone.repulsion_params['damping'],
                drone.repulsion_params['exploration_bias']
            ]
            actions.append(action)
        
        # Atualizar os drones com a lógica existente
        update_drones(drones, enemies)
        
        # Atualizar cobertura da grade
        update_grid_coverage(drones, grid_coverage)
        
        # Calcular recompensa global
        current_global_coverage = calculate_global_coverage(grid_coverage)
        current_global_detection = calculate_global_detection(drones)
        
        global_reward = calculate_global_reward(
            drones,
            prev_global_coverage,
            current_global_coverage,
            prev_global_detection,
            current_global_detection
        )
        
        # Extrair próximos estados
        next_states = [extract_state_for_drone(drone, drones, enemies, grid_coverage) for drone in drones]
        
        # Armazenar transição
        done = False  # Não importa muito para a inicialização
        maddpg_agent.store_transition(states, actions, global_reward, next_states, done)
        
        # Atualizar valores para próxima iteração
        prev_global_coverage = current_global_coverage
        prev_global_detection = current_global_detection
        
        samples_collected += 1
        
        if samples_collected % 100 == 0:
            print(f"Coletadas {samples_collected}/{num_samples} amostras para clonagem comportamental.")
    
    print("Inicialização com clonagem comportamental concluída!")
    
    # Treinar o agente por algumas iterações com os dados coletados
    for _ in range(50):
        maddpg_agent.train()


# Função para executar experimentos comparativos
def run_comparative_experiment():
    """
    Executa experimentos comparando o desempenho do enxame de drones:
    1. Usando apenas a estratégia de repulsão tradicional
    2. Usando a estratégia otimizada pelo MADDPG
    """
    # Configurações
    num_drones = 10
    num_trials = 10
    max_steps = 200
    
    # Métricas para comparação
    traditional_coverage = []
    traditional_detection = []
    traditional_energy = []
    
    maddpg_coverage = []
    maddpg_detection = []
    maddpg_energy = []
    
    # Inicializar o agente MADDPG
    state_dim = 30
    action_dim = 4
    action_bounds = (0.0, 5.0)
    
    state_dims = [state_dim] * num_drones
    action_dims = [action_dim] * num_drones
    maddpg_agent = GlobalRewardMADDPG(num_drones, state_dims, action_dims, action_bounds)
    
    # Carregar modelo treinado
    try:
        maddpg_agent.load_models("maddpg_global_models_final")
        print("Modelo pré-treinado carregado com sucesso!")
    except:
        print("Erro: Modelo pré-treinado não encontrado! Experimento comparativo requer um modelo treinado.")
        return
    
    # Executar experimentos
    for trial in range(num_trials):
        print(f"\nIniciando trial {trial+1}/{num_trials}")
        
        # Inicializar ambiente com as mesmas condições para ambos os métodos
        seed = trial * 100
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        # 1. Executar com estratégia tradicional
        print("Executando com estratégia tradicional...")
        drones_traditional = initialize_drones(num_drones)
        enemies = initialize_enemies()
        grid_coverage_traditional = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        total_energy_traditional = 0
        
        for step in range(max_steps):
            # Atualizar vizinhos
            update_drone_neighbors(drones_traditional)
            
            # Atualizar os drones com a lógica tradicional
            # (aqui você usa a estratégia de repulsão tradicional)
            update_drones_traditional(drones_traditional, enemies)
            
            # Calcular energia consumida
            for drone in drones_traditional:
                total_energy_traditional += drone.vel.x**2 + drone.vel.y**2
            
            # Atualizar cobertura da grade
            update_grid_coverage(drones_traditional, grid_coverage_traditional)
        
        # Calcular métricas finais
        final_coverage_traditional = calculate_global_coverage(grid_coverage_traditional)
        final_detection_traditional = calculate_global_detection(drones_traditional)
        
        traditional_coverage.append(final_coverage_traditional)
        traditional_detection.append(final_detection_traditional)
        traditional_energy.append(total_energy_traditional)
        
        # 2. Executar com estratégia MADDPG
        print("Executando com estratégia MADDPG...")
        # Reiniciar o seed para garantir as mesmas condições iniciais
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        
        drones_maddpg = initialize_drones(num_drones)
        enemies = initialize_enemies()  # Mesmos inimigos da execução anterior
        grid_coverage_maddpg = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        total_energy_maddpg = 0
        
        for step in range(max_steps):
            # Atualizar vizinhos
            update_drone_neighbors(drones_maddpg)
            
            # Extrair estados
            states = [extract_state_for_drone(drone, drones_maddpg, enemies, grid_coverage_maddpg) for drone in drones_maddpg]
            
            # Obter ações do MADDPG
            actions = maddpg_agent.get_actions(states, add_noise=False)
            
            # Aplicar ações aos drones
            for i, drone in enumerate(drones_maddpg):
                if drone.state == "HOLD_SPREAD":
                    apply_maddpg_actions(drone, actions[i])
            
            # Atualizar os drones
            update_drones_maddpg(drones_maddpg, enemies)
            
            # Calcular energia consumida
            for drone in drones_maddpg:
                total_energy_maddpg += drone.vel.x**2 + drone.vel.y**2
            
            # Atualizar cobertura da grade
            update_grid_coverage(drones_maddpg, grid_coverage_maddpg)
        
        # Calcular métricas finais
        final_coverage_maddpg = calculate_global_coverage(grid_coverage_maddpg)
        final_detection_maddpg = calculate_global_detection(drones_maddpg)
        
        maddpg_coverage.append(final_coverage_maddpg)
        maddpg_detection.append(final_detection_maddpg)
        maddpg_energy.append(total_energy_maddpg)
        
        # Imprimir resultados do trial
        print(f"Trial {trial+1} - Tradicional: Cobertura={final_coverage_traditional:.2f}, Detecção={final_detection_traditional}")
        print(f"Trial {trial+1} - MADDPG: Cobertura={final_coverage_maddpg:.2f}, Detecção={final_detection_maddpg}")
    
    # Calcular médias e desvios padrão
    trad_cov_mean = np.mean(traditional_coverage)
    trad_cov_std = np.std(traditional_coverage)
    trad_det_mean = np.mean(traditional_detection)
    trad_det_std = np.std(traditional_detection)
    trad_energy_mean = np.mean(traditional_energy)
    
    maddpg_cov_mean = np.mean(maddpg_coverage)
    maddpg_cov_std = np.std(maddpg_coverage)
    maddpg_det_mean = np.mean(maddpg_detection)
    maddpg_det_std = np.std(maddpg_detection)
    maddpg_energy_mean = np.mean(maddpg_energy)
    
    # Imprimir resultados comparativos
    print("\n==== RESULTADOS COMPARATIVOS ====")
    print(f"Estratégia Tradicional:")
    print(f"- Cobertura média: {trad_cov_mean:.4f} ± {trad_cov_std:.4f}")
    print(f"- Detecção média: {trad_det_mean:.1f} ± {trad_det_std:.1f}")
    print(f"- Energia total média: {trad_energy_mean:.1f}")
    
    print(f"\nEstratégia MADDPG:")
    print(f"- Cobertura média: {maddpg_cov_mean:.4f} ± {maddpg_cov_std:.4f}")
    print(f"- Detecção média: {maddpg_det_mean:.1f} ± {maddpg_det_std:.1f}")
    print(f"- Energia total média: {maddpg_energy_mean:.1f}")
    
    improvement_coverage = (maddpg_cov_mean - trad_cov_mean) / trad_cov_mean * 100
    improvement_detection = (maddpg_det_mean - trad_det_mean) / trad_det_mean * 100
    improvement_energy = (trad_energy_mean - maddpg_energy_mean) / trad_energy_mean * 100
    
    print(f"\nMelhoria com MADDPG:")
    print(f"- Cobertura: {improvement_coverage:.1f}%")
    print(f"- Detecção: {improvement_detection:.1f}%")
    print(f"- Eficiência energética: {improvement_energy:.1f}%")
    
    # Visualizar resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(['Tradicional', 'MADDPG'], [trad_cov_mean, maddpg_cov_mean], yerr=[trad_cov_std, maddpg_cov_std])
    plt.ylabel('Cobertura da Grade (%)')
    plt.title('Cobertura Média')
    
    plt.subplot(1, 3, 2)
    plt.bar(['Tradicional', 'MADDPG'], [trad_det_mean, maddpg_det_mean], yerr=[trad_det_std, maddpg_det_std])
    plt.ylabel('Células Detectadas')
    plt.title('Detecção Média')
    
    plt.subplot(1, 3, 3)
    plt.bar(['Tradicional', 'MADDPG'], [trad_energy_mean, maddpg_energy_mean])
    plt.ylabel('Energia Total')
    plt.title('Consumo de Energia')
    
    plt.tight_layout()
    plt.savefig('maddpg_vs_traditional_comparison.png')
    plt.show()


if __name__ == "__main__":
    # Aqui você pode escolher qual função executar
    
    # Para treinar um novo modelo:
    # train_maddpg_global_reward(num_episodes=1000, max_steps=200)
    
    # Para integrar com sua simulação existente:
    # integrate_with_simulation()
    
    # Para executar experimentos comparativos:
    # run_comparative_experiment()
    
    # Executar treinamento como exemplo
    train_maddpg_global_reward(num_episodes=500, max_steps=200)
