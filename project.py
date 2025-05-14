import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

period = 365
# Загрузка данных
ticker = "GAZP.ME"  
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Логарифмические доходности
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
returns = data['Log_Returns'].dropna()

def geometric_brownian_motion(S0, mu, sigma, T=1, N=period, n_simulations=5):
    """
    S0: начальная цена
    mu: средняя доходность (по лог. доходам)
    sigma: волатильность (среднеквадратичное отклонение лог. доходностей)
    T: период в годах
    N: количество временных шагов
    n_simulations: количество траекторий
    """
    dt = T/N
    t = np.linspace(0, T, N)
    S = np.zeros((N, n_simulations))
    S[0] = S0
    
    for i in range(1, N):
        Z = np.random.standard_normal(n_simulations)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return t, S

# Параметры из исторических данных
mu = returns.mean() * period  
sigma = returns.std() * np.sqrt(period) 
S0 = data['Close'].iloc[-1] 

# Моделирование
t, S = geometric_brownian_motion(S0, mu, sigma, T=1, N=period, n_simulations=10)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.title('Геометрическое броуновское движение (модель цены акции)')
plt.xlabel('Время (годы)')
plt.ylabel('Цена')
plt.grid(True)
plt.show()

# Реальные данные
plt.subplot(1, 2, 1)
plt.plot(data['Close'].values[-period:]) 
plt.title('Реальные данные')
plt.xlabel('День')
plt.ylabel('Цена')

# Смоделированные данные
plt.subplot(1, 2, 2)
t, S = geometric_brownian_motion(S0, mu, sigma, T=1, N=period, n_simulations=1)
plt.plot(S)
plt.title('Смоделированные данные (GBM)')
plt.xlabel('День')

plt.tight_layout()
plt.show()

#  Цепи Маркова
# Дискретизация доходностей на 3 состояния
n_states = 3
discretizer = KBinsDiscretizer(n_bins=n_states, encode='ordinal', strategy='quantile')
states = discretizer.fit_transform(returns.values.reshape(-1, 1)).flatten().astype(int)

# Построение матрицы переходов
transition_matrix = np.zeros((n_states, n_states))

for (i, j) in zip(states[:-1], states[1:]):
    transition_matrix[i, j] += 1

# Нормализация (сумма по строкам = 1)
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

print("Матрица переходов:")
print(transition_matrix)

# Моделирование цепи Маркова
def simulate_markov_chain(transition_matrix, n_steps=100, initial_state=None):
    n_states = transition_matrix.shape[0]
    if initial_state is None:
        initial_state = np.random.choice(n_states)
    
    states = [initial_state]
    for _ in range(n_steps-1):
        next_state = np.random.choice(n_states, p=transition_matrix[states[-1]])
        states.append(next_state)
    
    return states

# Симуляция
simulated_states = simulate_markov_chain(transition_matrix, n_steps=period, initial_state=1)

# Визуализация
plt.figure(figsize=(10, 4))
plt.plot(simulated_states, 'o-')
plt.title('Моделирование цепи Маркова (дискретные состояния доходностей)')
plt.xlabel('День')
plt.ylabel('Состояние')
plt.yticks(range(n_states), ['Сильное падение', 'Незначительное изменение', 'Рост'])
plt.grid(True)
plt.show()


