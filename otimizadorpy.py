import numpy as np
import sympy as sp
from scipy.linalg import solve
import re

def otimizar(funcao_str, variaveis_str, x0, metodo='newton', tol=1e-6, max_iter=100):
    # Criação das variáveis simbólicas
    var_names = variaveis_str.replace(' ', '').split(',')
    vars_sym = sp.symbols(var_names)
    f = sp.sympify(funcao_str)

    # Gradiente e Hessiana
    grad_f = sp.Matrix([sp.diff(f, var) for var in vars_sym])
    hess_f = sp.hessian(f, vars_sym)

    # Funções Numéricas (com lambdify para uso com NumPy)
    grad_f_func = sp.lambdify(vars_sym, grad_f, 'numpy')
    hess_f_func = sp.lambdify(vars_sym, hess_f, 'numpy')
    f_func = sp.lambdify(vars_sym, f, 'numpy')

    # Inicialização
    x_k = np.array(x0, dtype=float)
    I = np.eye(len(x_k))

    for k in range(max_iter):
        grad = np.array(grad_f_func(*x_k)).astype(np.float64).reshape(-1)
        hess = np.array(hess_f_func(*x_k), dtype=float)

        # Critério de parada baseado na norma do gradiente
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-6:
            print(f'\nConvergência em {k} iterações pelo critério: ||∇f(x^{k})|| < 1e-6')
            return x_k, f_func(*x_k)

        # Modificação da Hessiana se necessário
        if metodo == 'modificado':
            eigvals = np.linalg.eigvalsh(hess)
            min_eig = np.min(eigvals)
            if min_eig <= 0:
                tau = 1e-3 - min_eig
                hess += tau * I

        # Resolução do sistema linear H*p = -grad
        try:
            p_k = solve(hess, -grad)
        except np.linalg.LinAlgError:
            print('Hessiana singular. Método falhou.')
            return x_k, None

        # Atualização do ponto
        x_k = x_k + p_k

    print('\nMáximo de iterações atingido. Solução aproximada retornada.')
    return x_k, f_func(*x_k)

# =============================
#          Interface
# =============================
if __name__ == "__main__":
    print("Otimização com método de Newton / Newton Modificado")
    funcao_str = input("Digite a função a ser minimizada (ex: 100*(x2 - x1**2)**2 + (1 - x1)**2): ")
    variaveis_str = input("Digite as variáveis separadas por vírgula (ex: x1, x2): ")

    x0_str = input("Digite o ponto inicial como lista (ex: -1.0, 2.0): ")
    x0 = [float(num) for num in re.findall(r'-?\d+\.?\d*', x0_str)]

    metodo = input("Escolha o método ('newton' ou 'modificado'): ").strip().lower()

    # Execução
    xmin, fmin = otimizar(funcao_str, variaveis_str, x0, metodo)

    print("\nMínimo encontrado:")
    print("x* =", xmin)
    print("f(x*) =", fmin)
