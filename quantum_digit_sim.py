import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

class DigitalQuantumSystem:
    def __init__(self, q=2, n_minus=0, n_plus=3, d1=0.0):
        """
        Инициализация системы цифрового представления.
        
        Параметры:
        q: Основание системы счисления (radix).
        n_minus: Количество цифр после запятой (дробная часть).
        n_plus: Количество цифр перед запятой (целая часть).
        d1: Сдвиг наименьшей цифры (параметр решетки, аналог фазы).
        """
        self.q = q
        self.n_minus = n_minus
        self.n_plus = n_plus
        
        # Размерность гильбертова пространства N = q^(n_plus + n_minus)
        self.N = q ** (n_plus + n_minus)
        
        # Шаг решетки и период (исправляем формулы из статьи)
        # dx = q^(-n_minus)
        self.dx = q ** (-n_minus)
        # Период координаты Xi = N * dx = q^(n_plus)
        self.Xi = self.N * self.dx
        
        # Шаг импульса и период импульса
        # dp = 1 / Xi
        self.dp = 1.0 / self.Xi
        # Период импульса Pi = N * dp = q^(n_minus)
        self.Pi = self.N * self.dp
        
        self.d1 = d1
        
        # Базис координаты (с учетом сдвига d1)
        # x_k = (k + shift_offset) * dx, где shift_offset зависит от d1
        # В статье сдвиг решетки определяется как -d1 * dx для наименьшего числа.
        # Для простоты реализуем сдвиг фазы в граничных условиях или сдвиг сетки.
        # Здесь мы сдвигаем саму сетку координат для наглядности эффекта d1.
        # Минимальное значение на решетке согласно ст. (25): x_min = -d1 * dx (приближенно для интерпретации)
        # Но строго: узлы x_k = k * dx. Сдвиг d1 влияет на определение цифр.
        
        self.x_grid = np.arange(self.N) * self.dx
        self.p_grid = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi # В единицах hbar=1, но статья использует 2pi нормировку
        
        # Корректировка сетки импульса под определения статьи (циклическая)
        # p values: 0, dp, 2dp ... (N/2)dp, -(N/2-1)dp ... -dp
        self.p_grid = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx)) * (2 * np.pi) 
        # Примечание: В статье используется нормировка где exp(2pi i A p). 
        # Будем использовать стандартный DFT подход, где p сопряжен x.

    def get_digit_operator(self, s, observable='x'):
        """
        Создает оператор s-й цифры для координаты ('x') или импульса ('p').
        Формула: digit_s = floor( (obs / q^s) ) mod q
        
        Для сдвинутой решетки (d1 != 0) формула модифицируется.
        """
        diag_values = np.zeros(self.N, dtype=int)
        
        if observable == 'x':
            # Значения координаты
            vals = self.x_grid
            
            # Учет сдвига d1: 
            # Согласно статье, сдвиг d1 меняет диапазон цифр {d1, d1+1, ...}
            # Реализуем через сдвиг аргумента перед взятием цифры
            shifted_vals = vals + self.d1 * self.dx 
            
            for i, x in enumerate(shifted_vals):
                # Вычисляем s-ю цифру: floor(x / q^s) % q
                # Для отрицательных чисел в Python floor работает корректно для нашей задачи
                power = self.q ** s
                digit = int(np.floor(x / power)) % self.q
                diag_values[i] = digit
                
        elif observable == 'p':
            # Аналогично для импульса (требуется осторожность с цикличностью)
            # В статье импульс тоже раскладывается в ряд.
            # Используем сетку импульсов от 0 до Pi
            vals = np.arange(self.N) * self.dp
            shifted_vals = vals + self.d1 * self.dp # Гипотетический сдвиг для импульса
            
            for i, p in enumerate(shifted_vals):
                power = self.q ** s
                digit = int(np.floor(p / power)) % self.q
                diag_values[i] = digit
        else:
            raise ValueError("Observable must be 'x' or 'p'")
            
        return np.diag(diag_values)

    def momentum_operator(self):
        """
        Создает оператор импульса p_hat.
        В цифровом представлении p_hat генерирует сдвиги по координате.
        p_hat = F^-1 * diag(p_values) * F
        """
        # Собственные значения импульса на решетке (от 0 до N-1) * dp
        # Чтобы получить эрмитов оператор с правильным спектром в диапазоне [-Pi/2, Pi/2]
        p_vals = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        # Сдвигаем частоты для корректного представления (optional, зависит от базиса)
        # Используем стандартный подход: p_diag = diag(2*pi*k/N/dx)
        
        P_diag = np.diag(p_vals)
        
        # Оператор сдвига (Фурье трансформация)
        # В статье T_A = exp(2pi i A p). Здесь мы строим сам p.
        F = np.fft.fft(np.eye(self.N), axis=0) / np.sqrt(self.N) # Нормированный DFT
        F_dag = np.conj(F.T)
        
        # p_hat = F_dag * P_diag * F
        p_hat = F_dag @ P_diag @ F
        
        return p_hat

    def position_operator(self):
        """Оператор координаты (диагональный в базисе x)"""
        return np.diag(self.x_grid)

    def create_gaussian_wavepacket(self, x0, p0, sigma):
        """Создает гауссов волновой пакет"""
        x = self.x_grid
        psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x)
        psi /= np.linalg.norm(psi) # Нормировка
        return psi

    def measure_digit_distribution(self, psi, s, observable='x'):
        """
        Измеряет распределение вероятностей для s-й цифры.
        Возвращает словарь {digit_value: probability}
        """
        D_s = self.get_digit_operator(s, observable)
        # Собственные значения диагональной матрицы - это значения цифр
        digits = np.diag(D_s).astype(int)
        
        probs = np.abs(psi)**2
        distribution = {}
        for d in range(self.q):
            mask = (digits == d)
            distribution[d] = np.sum(probs[mask])
            
        return distribution

def run_simulation():
    print("=== Инициализация квантовой системы цифр ===")
    # Параметры: троичная система (q=3), 1 цифра после запятой, 2 перед запятой
    q = 3
    n_minus = 1
    n_plus = 2
    d1 = 0.0 # Попробуйте изменить на -0.5 для симметричной системы
    
    system = DigitalQuantumSystem(q=q, n_minus=n_minus, n_plus=n_plus, d1=d1)
    
    print(f"Основание q: {q}")
    print(f"Размерность N: {system.N}")
    print(f"Шаг координаты dx: {system.dx:.4f}")
    print(f"Период координаты Xi: {system.Xi:.4f}")
    print(f"Сдвиг решетки d1: {d1}")

    # Создаем волновой пакет
    x0 = system.Xi / 2  # Центр ящика
    p0 = 0.0
    sigma = system.dx * 2.0 # Ширина пакета
    
    psi = system.create_gaussian_wavepacket(x0, p0, sigma)
    
    # 1. Визуализация волновой функции
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(system.x_grid, np.abs(psi)**2, 'b-', label='Prob Density')
    plt.plot(system.x_grid, np.real(psi), 'r--', alpha=0.5, label='Re(Psi)')
    plt.title(f"Волновая функция (q={q}, d1={d1})")
    plt.xlabel("Координата x")
    plt.ylabel("Вероятность / Амплитуда")
    plt.legend()
    plt.grid(True)

    # 2. Измерение цифр (Digit Measurement)
    # Проверим распределение для разных разрядов s
    digits_probs = []
    s_values = list(range(-n_minus, n_plus))
    
    for s in s_values:
        dist = system.measure_digit_distribution(psi, s, observable='x')
        probs = [dist.get(d, 0) for d in range(q)]
        digits_probs.append(probs)
        
    plt.subplot(1, 2, 2)
    width = 0.8 / len(s_values)
    x_axis = np.arange(q)
    
    for i, s in enumerate(s_values):
        offset = (i - len(s_values)/2) * width
        plt.bar(x_axis + offset, digits_probs[i], width, label=f's={s}')
        
    plt.title("Распределение вероятностей по цифрам разрядов")
    plt.xlabel("Значение цифры (0..q-1)")
    plt.ylabel("Вероятность")
    plt.xticks(np.arange(q))
    plt.legend(title="Разряд s")
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

    # 3. Демонстрация нелокальности импульса
    print("\n=== Анализ оператора импульса ===")
    p_hat = system.momentum_operator()
    
    # Проверим коммутатор [x, p] (должен быть близок к i, но с поправками на решетку)
    x_hat = system.position_operator()
    comm = x_hat @ p_hat - p_hat @ x_hat
    
    # Диагональный элемент коммутатора (должен быть ~ i)
    # В дискретном случае это сложно, посмотрим на след или диагональ
    diag_comm = np.diag(comm)
    print(f"Среднее значение диагонали коммутатора [x, p]: {np.mean(diag_comm):.4f}i (ожидается ~1.0i)")
    
    # Проверка нелокальности: сколько ненулевых элементов в строке p_hat?
    non_zero_counts = np.count_nonzero(np.abs(p_hat) > 1e-10, axis=1)
    print(f"Количество ненулевых элементов в строках p_hat (нелокальность): {non_zero_counts[0]} из {system.N}")
    print("Вывод: Оператор импульса действительно нелокален (заполнен почти полностью).")

    # 4. Эффект сдвига d1 (Ааронов-Бом аналогия)
    if d1 == 0:
        print("\nЗапустите симуляцию снова с d1=-0.5, чтобы увидеть эффект сдвига решетки (симметричная система).")

if __name__ == "__main__":
    run_simulation()
