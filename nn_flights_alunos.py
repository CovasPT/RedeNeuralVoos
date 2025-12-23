import random
import math
import matplotlib.pyplot as plt

alpha = 0.2 #velocidade de aprendizagem
carriers = ["AA", "B6", "DL", "UA", "WN"]
aeroportos = ["ATL", "DEN", "JFK", "LAX", "MIA", "ORD", "SFO", "SEA"]
#------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(inp):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(-inp))


def forward(nn, input):
    nn['x'] = input.copy()
    nn['x'].append(-1) # Bias entrada

    # Calcula Camada Escondida (Z) - Uma linha faz tudo!
    nn['z'] = [sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1) # Bias escondido

    # Calcula Saída (Y)
    nn['y'] = [sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [ [w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [ [w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    forward(nn, input)
    error(nn, output)
    update(nn)
    #print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    

#-----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net


#-------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR VOOS DE AVIÃO---------    

"""Funcao principal do nosso programa para prever se um voo chegará com ou sem atraso:
cria os conjuntos de treino e teste, chama a funcao que cria e treina a rede e, por fim, 
a funcao que a testa. A funcao recebe como argumento o ficheiro correspondente ao dataset 
que deve ser usado, os tamanhos das camadas de entrada, escondida e saída,
o numero de epocas que deve ser considerado no treino e os tamanhos dos conjuntos de treino e 
teste"""
def run_fights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size):
    # 1. Criar os conjuntos de dados
    print("A carregar e a processar o dataset...")
    treino, teste = build_sets(file, training_set_size, test_set_size)
    
    # 2. Treinar a rede
    print(f"A iniciar treino por {epochs} épocas...")
   #Escreve isto (o '_' serve para ignorar o histórico, que não precisas aqui):
    rede_treinada, _ = train_flights(input_size, hidden_size, output_size, treino, teste, epochs)
    
    # 3. Avaliação final
    print("\n--- RESULTADOS FINAIS NO CONJUNTO DE TESTE ---") 
    
    # CORREÇÃO: Executar a função primeiro e capturar o retorno em 'resultados'
    # Nota: Se o 'printing=True' já faz prints lá dentro, podes querer mudar para False aqui para não duplicar, 
    # ou manter True se queres ver o output da função também.
    resultados = test_flights(rede_treinada, teste, printing=False)

    print("\nResumo do Desempenho:")
    # Agora a variável 'resultados' já existe e pode ser usada
    print(f"Exatidão: {resultados['exatidão']:.2f}%")
    print(f"Precisão: {resultados['precisão']:.2f}")
    print(f"Cobertura (Recall): {resultados['cobertura']:.2f}")
    print(f"F1-Score: {resultados['f1-score']:.2f}")
    print(f"Matriz de Confusão [VP, FP, VN, FN]: {resultados['matriz']}")
    
    return resultados

"""Funcao que cria os conjuntos de treino e de teste a partir dos dados
armazenados em f ('DataSet1.csv'). A funcao le cada linha, 
tranforma-a numa lista de valores e chama a funcao translate para a colocar no 
formato adequado para o padrao de treino. Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de 
treino --->x e o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº 
de estudantes disponível no dataset"""       
def build_sets(nome_f, x, y):
    all_patterns = []
    
    with open(nome_f, 'r') as f:
        linhas = f.readlines()
        # Começamos no 1 para saltar o cabeçalho (Month, DayOfWeek...)
        for i in range(1, len(linhas)):
            # 1. Limpa a linha e separa por vírgulas
            colunas = linhas[i].strip().split(',')
            
            # 2. CORREÇÃO: Passar 'colunas' (a linha atual) e não 'all_patterns'
            padrao = translate(colunas)
            
            # 3. Adiciona o padrão processado à lista global
            all_patterns.append(padrao)

    # Só baralhamos e dividimos DEPOIS de processar todas as linhas
    random.shuffle(all_patterns)
    
    treino = all_patterns[:x]
    teste = all_patterns[x:x+y]
    
    return treino, teste

"""A função translate recebe cada lista de valores que caracterizam um voo
e transforma-a num padrão de treino. Cada padrão é uma lista com o seguinte formato 
[padrao_de_entrada, classe_do_voo, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""
def translate(lista):
    mes_norm = normaliza_valores(float(lista[0]), 1, 12)
    # 2. One-hot da Companhia (UniqueCarrier está no índice 3) [cite: 69, 177]
    carrier_onehot = converte_categ_numerico(lista[3], carriers)
    
    # 3. One-hot da Origem e Destino 
    origem_onehot = converte_categ_numerico(lista[4], aeroportos)
    destino_onehot = converte_categ_numerico(lista[5], aeroportos)
    
    # ... e por aí fora para os restantes atributos ...
    # No final, juntas (concatenas) as listas todas com o símbolo +
    entrada = [mes_norm] + carrier_onehot + origem_onehot + destino_onehot
    
    # Classe real (Delayed está no fim) 
    atraso = int(lista[7])
    saida_rede = [1, 0] if atraso == 0 else [0, 1]
    
    return [entrada, atraso, saida_rede]
    pass

#Função que converte valores categóricos para a codificação onehot                
def converte_categ_numerico(instancia, categorias_possiveis):
   # Criamos uma lista de zeros com o tamanho das categorias existentes
    vetor = [0] * len(categorias_possiveis)
    
    # Se o valor existir na nossa lista, pomos 1 na posição dele
    if instancia in categorias_possiveis:
        indice = categorias_possiveis.index(instancia)
        vetor[indice] = 1
        
    return vetor


"""Função que normaliza os valores necessários"""   
def normaliza_valores(valor, min, max):
    return (valor - min) / (max - min)
    #A definir pelos estudantes
    pass
       


"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento 
o conjunto de treino, os tamanhos das camadas de entrada, escondida e saída e o número 
de épocas que irão ser usadas para fazer o treino"""
def train_flights(input_size, hidden_size, output_size, training_set, test_set, epochs, plot=True):
    net = make(input_size, hidden_size, output_size)
    
    historico_treino = []
    historico_teste = []

    # Print inicial para saberes que começou
    print(f"-> A iniciar treino: {hidden_size} neurónios, {epochs} épocas.")

    for e in range(epochs):
        random.shuffle(training_set)
        
        for padrao in training_set:
            # NOTA: Comenta o print dentro da função 'iterate' original para isto ser rápido!
            iterate(e, net, padrao[0], padrao[2])
            
        # Avaliação silenciosa a cada época
        res_treino = test_flights(net, training_set, printing=False)
        res_teste = test_flights(net, test_set, printing=False)
        
        historico_treino.append(res_treino['exatidão'])
        historico_teste.append(res_teste['exatidão'])
        
        # O QUE FALTAVA: Feedback visual a cada época
        print(f"   Época {e+1:02d}/{epochs} | Treino: {res_treino['exatidão']:.1f}% | Teste: {res_teste['exatidão']:.1f}%")
        
    if plot:
        plt.figure()
        plt.plot(historico_treino, label="Treino")
        plt.plot(historico_teste, label="Teste")
        plt.title(f'Evolução com {hidden_size} Neurónios')
        plt.xlabel('Épocas')
        plt.ylabel('Exatidão (%)')
        plt.legend()
        plt.show()
    
    # Retornamos os históricos para poder comparar nas experiências
    return net, historico_teste


"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste ou treino.
Para cada padrao do conjunto chama a funcao forward e determina a classe do voo
que corresponde ao maior valor da lista de saida. A classe determinada pela rede
deve ser comparada com a classe real,sendo contabilizado o número de respostas corretas. 
A função calcula a percentagem de respostas corretas, 
o nº de VP,FP,VN, FN, precisão, cobertura e f1-score""" 
def test_flights(net, test_set, printing = True):
    vp, fp, vn, fn = 0, 0, 0, 0
    
    for i in range(len(test_set)):
        padrao = test_set[i]
        entrada = padrao[0]
        classe_real = padrao[1]
        
        # Faz a propagação para a frente
        forward(net, entrada)
        # Obtém a previsão da rede
        previsao = retranslate(net['y'])
        
        if printing:
            print(f"Voo {i}: Rede previu {previsao}, Realidade era {classe_real}")
            
        # Contagem para a matriz de confusão
        if previsao == 1 and classe_real == 1: vp += 1
        elif previsao == 1 and classe_real == 0: fp += 1
        elif previsao == 0 and classe_real == 0: vn += 1
        elif previsao == 0 and classe_real == 1: fn += 1

    # Cálculos das métricas (evitando divisões por zero)
    exatidao = (vp + vn) / len(test_set) * 100
    # Precisão
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    # Cobertura (Recall)
    cobertura = vp / (vp + fn) if (vp + fn) > 0 else 0
    # F1-Score
    f1 = 2 * (precisao * cobertura) / (precisao + cobertura) if (precisao + cobertura) > 0 else 0
    
    if printing:
        print(f"Success rate: {exatidao:.2f}%")
        # Imprime os restantes valores aqui
    return {"exatidão": exatidao, "matriz": [vp, fp, vn, fn], "precisão": precisao, "cobertura": cobertura, "f1-score": f1} 
    pass
  
"""Recebe o padrao de saida da rede e devolve a situação de atraso do voo.
A situação de atraso corresponde ao indice da saida com maior valor."""  
def retranslate(out):
    if out[0] > out[1]:
        return 0
    else:
        return 1
    pass


def experiencia_neuronios(file, training_size, test_size):
    # --- CONFIGURAÇÕES (MANTER) ---
    input_size = 25 
    output_size = 2
    epochs = 50
    neuronios_para_testar = [4, 8, 12, 20]
    
    treino, teste = build_sets(file, training_size, test_size)
    
    resultados_grafico = {} # Para o gráfico (histórico)
    resultados_tabela = []  # NOVO: Para guardar os números finais (VP, FP, etc.)

    print("\n=== INÍCIO DA EXPERIÊNCIA 8.2.1: VARIAÇÃO DE NEURÓNIOS ===")
    
    for n in neuronios_para_testar:
        print(f"\n[Configuração] Testando com {n} neurónios...")
        
        global alpha
        alpha = 0.01 
        
        # ALTERAÇÃO 1: Guardar a 'rede' em vez de usar '_'
        rede_treinada, historico = train_flights(input_size, n, output_size, treino, teste, epochs, plot=False)
        
        # Guardar histórico para o gráfico
        resultados_grafico[n] = historico
        
        # NOVO (ALTERAÇÃO 2): Calcular as métricas finais desta rede
        # Isto vai devolver o tal dicionário com {'matriz': ..., 'exatidão': ...}
        stats = test_flights(rede_treinada, teste, printing=False)
        
        # Guardamos o número de neurónios e as stats numa lista
        resultados_tabela.append((n, stats))

    # --- PARTE 1: TABELA DE RESULTADOS (NOVO) ---
    print("\n" + "="*80)
    print(f"{'Neurónios':<10} | {'Exatidão':<10} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Matriz [VP,FP,VN,FN]':<20}")
    print("-" * 80)
    
    for n, dados in resultados_tabela:
        # Aqui extraímos os valores do dicionário 'stats'
        acc = dados['exatidão']
        prec = dados['precisão']
        rec = dados['cobertura'] # ou 'recall', depende do nome que deste no dicionário
        f1 = dados['f1-score']
        mat = dados['matriz']
        
        print(f"{n:<10} | {acc:.2f}%     | {prec:.2f}       | {rec:.2f}       | {f1:.2f}       | {mat}")
    print("="*80)

    # --- PARTE 2: GRÁFICO (MANTER) ---
    print("\n-> A gerar gráfico comparativo...")
    plt.figure(figsize=(10, 6))
    
    for n in neuronios_para_testar:
        plt.plot(resultados_grafico[n], label=f'{n} Neurónios')
        
    plt.title('Comparação: Variação de Neurónios na Camada Escondida (Teste)')
    plt.xlabel('Épocas')
    plt.ylabel('Exatidão no Teste (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Experiência concluída.")



def experiencia_alpha(file, training_size, test_size, melhor_n_neuronios):
    # --- CONFIGURAÇÕES ---
    # Nota: Aqui o input_size e epochs mantêm-se constantes
    input_size = 25 
    output_size = 2
    epochs = 50 # Ou outro valor se o enunciado pedir
    
    # Valores de Alpha a testar (confirma no teu PDF quais são os valores pedidos)
    alphas_para_testar = [0.01, 0.1, 0.5, 0.9] 
    
    treino, teste = build_sets(file, training_size, test_size)
    
    resultados_grafico = {}
    resultados_tabela = []

    print(f"\n=== EXPERIÊNCIA 8.2.2: VARIAÇÃO DE ALPHA (com {melhor_n_neuronios} Neurónios) ===")
    
    for val_alpha in alphas_para_testar:
        print(f"\n[Configuração] Testando com Alpha = {val_alpha}...")
        
        # 1. ATUALIZAR A VARIÁVEL GLOBAL ALPHA
        # Isto é perigoso, mas como o teu código usa 'global alpha', tem de ser assim:
        global alpha
        alpha = val_alpha
        
        # 2. TREINAR
        # CUIDADO: Tens de garantir que passas o 'melhor_n_neuronios' aqui
        rede_treinada, historico = train_flights(input_size, melhor_n_neuronios, output_size, treino, teste, epochs, plot=False)
        
        resultados_grafico[val_alpha] = historico
        
        # 3. TESTAR
        stats = test_flights(rede_treinada, teste, printing=False)
        resultados_tabela.append((val_alpha, stats))

    # --- APRESENTAÇÃO DE RESULTADOS ---
    # Copia a lógica da tabela que fizemos antes, mas muda a coluna 'Neurónios' para 'Alpha'
    print("\n" + "="*80)
    print(f"{'Alpha':<10} | {'Exatidão':<10} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    
    for a, dados in resultados_tabela:
        print(f"{a:<10} | {dados['exatidão']:.2f}%     | {dados['precisão']:.2f}       | {dados['cobertura']:.2f}       | {dados['f1-score']:.2f}")
    
    # GERA O GRÁFICO (Adaptado para Alphas)
    # ... (O teu código de plot aqui, usando 'resultados_grafico')
    plt.figure(figsize=(10, 6))

    for val_alpha in alphas_para_testar:
        plt.plot(resultados_grafico[val_alpha], label=f'Alpha = {val_alpha}')
        
    plt.title(f'Comparação: Variação de Alpha (Teste) - {melhor_n_neuronios} Neurónios')
    plt.xlabel('Épocas')
    plt.ylabel('Exatidão no Teste (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Experiência concluída.")

# --- COMO CORRER ---
if __name__ == "__main__":
    # Para correr a experiência dos neurónios:
    experiencia_neuronios('DataSet1.csv', 800, 200)

if __name__ == "__main__":
    #Vamos treinar durante 1000 épocas uma rede para aprender a função logica AND
    #Faz testes para números de épocas diferentes e para as restantes funções lógicas já implementadas
    rede_AND = train_and(1000)
    #Agora vamos ver se ela aprendeu bem
    tabela_verdade = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
    for linha in tabela_verdade:
        forward(rede_AND, list(linha))
        print('A rede determinou %s para a entrada %d AND %d quando devia ser %d'
              %(rede_AND['y'], linha[0], linha[1], tabela_verdade[linha]))
        

if __name__ == "__main__":
    # Parâmetros base do enunciado
    # input_size depende do teu One-Hot (Ex: 4 numéricos + 5 carriers + 8 origens + 8 destinos = 25)
    # Ajusta este valor de acordo com o tamanho final do teu vetor no translate
    input_size = 25 
    hidden_size = 8
    output_size = 2
    epochs = 50
    training_set_size = 800
    test_set_size = 200
    file = 'DataSet1.csv'
    
    run_fights(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size)