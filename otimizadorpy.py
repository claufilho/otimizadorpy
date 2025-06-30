import numpy as np
import sympy as sp
from scipy.linalg import solve, LinAlgError
import re

def otimizar(funcao_str, variaveis_str, x0, metodo='newton', tol=1e-6, max_iter=100):
    # Criação das variáveis
    var_names = variaveis_str.replace(' ', '').split(',')
    vars_sym = sp.symbols(var_names)
    f = sp.sympify(funcao_str)

    # Gradiente e Hessiana
    grad_f = sp.Matrix([sp.diff(f, var) for var in vars_sym])
    hess_f = sp.hessian(f, vars_sym)

    # Funções Numéricas
    grad_f_func = sp.lambdify(vars_sym, grad_f, 'numpy')
    hess_f_func = sp.lambdify(vars_sym, hess_f, 'numpy')
    f_func = sp.lambdify(vars_sym, f, 'numpy')

    # Início do método
    x_k = np.array(x0, dtype=float)
    I = np.eye(len(x_k))

    for k in range(max_iter):
        # Cálculo do gradiente e Hessiana na posição atual
        grad = np.array(grad_f_func(*x_k)).astype(np.float64).reshape(-1)
        hess = np.array(hess_f_func(*x_k), dtype=float)

        # Verificação do critério de parada baseado na norma do gradiente
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f'\n✅ Convergência atingida em {k} iterações. Norma do gradiente: {grad_norm:.6e}')
            return x_k, f_func(*x_k)

        # Verificando o critério de convergência para a função objetivo
        f_value = f_func(*x_k)
        if abs(f_value) < tol:  # Se a função atingir um valor pequeno, pode ser considerado convergido
            print(f'\n✅ Convergência atingida em {k} iterações. Função objetivo: {f_value:.6e}')
            return x_k, f_value

        # Newton Modificado (caso a Hessiana não seja bem condicionada)
        if metodo == 'modificado':
            eigvals = np.linalg.eigvalsh(hess)
            min_eig = np.min(eigvals)
            if min_eig <= 0:
                tau = 1e-3 - min_eig  # Ajuste da regularização para melhorar a condição da Hessiana
                hess += tau * I

        try:
            # Cálculo do passo de Newton
            p_k = solve(hess, -grad)
        except LinAlgError:
            print('❌ Hessiana singular. Método falhou.')
            return x_k, None

        # Atualização da posição
        x_k = x_k + p_k

        # Exibindo o progresso
        if k % 10 == 0:  # A cada 10 iterações, mostramos o progresso
            print(f'Iteração {k}: Norma do gradiente = {grad_norm:.6e}, Função objetivo = {f_value:.6e}')

    print('\n⚠️ Máximo de iterações atingido.')
    return x_k, f_func(*x_k)

# =============================
#      Entrada do usuário
# =============================
if name == "_main": 
    print("🔧 Otimização com método de Newton / Newton Modificado")
    funcao_str = input("Digite a função a ser minimizada (ex: 100*(x2 - x1*2)2 + (1 - x1)*2): ")
    variaveis_str = input("Digite as variáveis separadas por vírgula (ex: x1, x2): ")

    x0_str = input("Digite o ponto inicial como lista (ex: -1.0, 2.0): ")
    x0 = [float(num) for num in re.findall(r'-?\d+\.?\d*', x0_str)]

    metodo = input("Escolha o método ('newton' ou 'modificado'): ").strip().lower()

    # Execução
    xmin, fmin = otimizar(funcao_str, variaveis_str, x0, metodo)

    print("\n📍 Mínimo encontrado:")
    print("x* =", xmin)
    print("f(x*) =", fmin)
