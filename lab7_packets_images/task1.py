"""
Тема 7. Вейвлет-пакеты. Задание 1. Наилучший базис для сжатия сигнала.

Сигналы на [0; 4), 1024 точки:
  a) cos(2πt) + 4cos(6πt) + r(t),  r — равномерный шум на [-0.5; 0.5]
  b) 4t - t^2 + r(t),              r — нормальный шум (0; 0.5)
Вейвлет db1 (это то же самое, что haar), 3 уровня разложения.

Что такое «наилучший базис»:
  В полном дереве вейвлет-пакета есть очень много способов выбрать набор
  узлов, из которых можно восстановить сигнал. Например, {a, d},
  или {aa, ad, d}, или {aaa, aad, ad, da, dd} и т. д. Главное — чтобы
  выбранные узлы вместе покрывали весь сигнал без пересечений.
  Каждый такой набор — это «базис».
  Для сжатия лучше тот базис, в котором энергия сигнала собрана в малое число
  больших коэффициентов, а остальные почти нули.

Как его выбирают (алгоритм Койфмана–Викерхаузера):
  1) каждому узлу приписываем «стоимость» — энтропию Шеннона
         H = -Σ p · ln(p),   p = c² / E   (E — энергия всего сигнала, Σ x²),
     p — доля энергии сигнала, которая приходится на данный коэффициент.
     Она мала, когда энергия сосредоточена в немногих коэффициентах,
     и велика, когда энергия «размазана» по многим;
  2) идём снизу вверх: если стоимость родителя меньше суммы стоимостей двух его
     детей (в их наилучшем варианте), оставляем родителя, иначе — детей.
  Получается дерево, которое на разных ветках может быть разной глубины.

Сжатие: в наилучшем базисе оставляем 50% коэффициентов (как в параграфе 4,
где из одноуровневого DWT выбрасывались все детали, то есть тоже половина)
и восстанавливаем сигнал. Затем сравниваем погрешности.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pywt

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

WAVELET = 'db1'
LEVELS = 3
KEEP = 0.5

# Сигналы генерируем точно так же, как в теме 4 (тот же seed и порядок),
# чтобы шум совпал и сравнение было честным.
np.random.seed(4321)
N = 1024
t = np.linspace(0, 4, N, endpoint=False)
SIGNALS = {
    'a': ("cos(2πt) + 4cos(6πt) + равномерный шум",
          np.cos(2*np.pi*t) + 4*np.cos(6*np.pi*t) + np.random.uniform(-0.5, 0.5, N)),
    'b': ("4t - t² + нормальный шум",
          4*t - t**2 + np.random.normal(0, 0.5, N)),
}


def entropy(c, total_energy):
    """Энтропия Шеннона: -Σ p ln p, где p = c² / (энергия всего сигнала).

    p — доля энергии, приходящаяся на коэффициент. Делим на энергию ВСЕГО
    сигнала (одно и то же число для всех узлов), поэтому стоимости узлов
    можно складывать и сравнивать. Считаем 0·ln 0 = 0.
    """
    p = c[c != 0]**2 / total_energy
    return -np.sum(p * np.log(p))


def best_basis(wp, energy, path=''):
    """Возвращает (список путей наилучшего базиса, его стоимость).

    Рекурсия: для узла сравниваем его собственную стоимость со стоимостью
    лучших базисов его детей path+'a' и path+'d'.
    """
    node = wp[path] if path else wp
    own_cost = entropy(np.asarray(node.data), energy)
    if len(path) == wp.maxlevel:           # лист — делить дальше некуда
        return [path], own_cost
    basis_a, cost_a = best_basis(wp, energy, path + 'a')
    basis_d, cost_d = best_basis(wp, energy, path + 'd')
    if own_cost <= cost_a + cost_d:
        return [path], own_cost
    return basis_a + basis_d, cost_a + cost_d


def plot_levels(wp, x, title, filename):
    """Как в методичке: сверху сигнал, ниже все узлы 1-го, 2-го, 3-го уровней."""
    rows = LEVELS + 1
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title)
    ax = fig.add_subplot(rows, 1, 1)
    ax.plot(x, lw=0.8)
    ax.set_title('исходный сигнал', fontsize=9)
    for level in range(1, LEVELS + 1):
        nodes = wp.get_level(level, 'natural')
        cols = len(nodes)
        for i, node in enumerate(nodes):
            ax = fig.add_subplot(rows, cols, level*cols + i + 1)
            ax.plot(node.data, lw=0.8)
            ax.set_title(node.path, fontsize=9)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / filename)
    plt.show()


for key, (title, x) in SIGNALS.items():
    print(f"\n=== Сигнал {key}: {title}")

    # Разложение в вейвлет-пакет на 3 уровня
    wp = pywt.WaveletPacket(data=x, wavelet=WAVELET, mode='symmetric', maxlevel=LEVELS)
    for level in range(1, LEVELS + 1):
        print(f"   уровень {level}:", [n.path for n in wp.get_level(level, 'natural')])

    # Изображения разложения на каждом уровне
    plot_levels(wp, x, f"Сигнал {key}: разложение в вейвлет-пакет ({WAVELET}, {LEVELS} уровня)",
                f"task1_signal_{key}_levels.png")

    # Наилучший базис
    energy = np.sum(x**2)        # энергия сигнала (у Хаара она сохраняется в каждом уровне)
    basis, cost = best_basis(wp, energy)
    print(f"   наилучший базис: {basis}  (энтропия {cost:.3f}, "
          f"у исходного сигнала {entropy(x, energy):.3f})")

    # Сжатие в наилучшем базисе: общий порог по всем узлам базиса,
    # оставляем KEEP самых больших по модулю коэффициентов.
    all_coefs = np.concatenate([wp[p].data for p in basis])
    thr = np.quantile(np.abs(all_coefs), 1 - KEEP)
    new_wp = pywt.WaveletPacket(data=None, wavelet=WAVELET, mode='symmetric')
    kept = 0
    for p in basis:
        c = np.where(np.abs(wp[p].data) >= thr, wp[p].data, 0)
        kept += np.count_nonzero(c)
        new_wp[p] = c
    x_rec = new_wp.reconstruct(update=False)[:N]
    err_packet = np.max(np.abs(x - x_rec))

    # Для сравнения — метод параграфа 4: одноуровневое DWT тем же вейвлетом,
    # детали обнулены (тоже остаётся половина коэффициентов).
    cA, cD = pywt.dwt(x, WAVELET)
    x_dwt = pywt.idwt(cA, np.zeros_like(cD), WAVELET)[:N]
    err_dwt = np.max(np.abs(x - x_dwt))

    print(f"   вейвлет-пакет, наилучший базис: оставлено {kept} коэф., "
          f"max погрешность = {err_packet:.4f}")
    print(f"   параграф 4 (DWT, cD = 0):       оставлено {len(cA)} коэф., "
          f"max погрешность = {err_dwt:.4f}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, x, 'r', lw=0.8, label='исходный')
    ax[0].plot(t, x_rec, 'g', lw=1, label='восстановленный')
    ax[0].set_title(f"Вейвлет-пакет, базис {basis}: {kept} коэф., max погрешность {err_packet:.3f}")
    ax[0].legend(loc='upper right')
    ax[1].plot(t, x, 'r', lw=0.8, label='исходный')
    ax[1].plot(t, x_dwt, 'b', lw=1, label='DWT, детали обнулены')
    ax[1].set_title(f"Параграф 4: одноуровневое DWT, {len(cA)} коэф., max погрешность {err_dwt:.3f}")
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('t')
    fig.tight_layout()
    fig.savefig(RESULTS / f"task1_signal_{key}_compare.png")
    plt.show()

# Как понимать результат (max погрешность при 50% коэффициентов):
#     сигнал a: вейвлет-пакет 0.44, параграф 4 — 0.63
#     сигнал b: вейвлет-пакет 0.47, параграф 4 — 1.16
#   * В пакете раскладываются и детали, поэтому разбиение частот можно
#     подобрать под сигнал. Для a наилучшим оказался весь 3-й уровень,
#     для b — 3-й уровень везде, кроме узла 'dd' (самые высокие частоты,
#     там только шум — дробить его бессмысленно, энтропия от этого не падает).
#   * Энтропия в наилучшем базисе (~4.7) заметно меньше, чем у исходного
#     сигнала (~6.6): энергия собралась в меньшем числе коэффициентов.
#   * При одинаковом числе коэффициентов (512) DWT из параграфа 4 обязан
#     оставить ВСЮ аппроксимацию и выбросить ВСЕ детали, а пакет оставляет
#     512 самых больших коэффициентов из любых частотных полос.
#     Поэтому погрешность меньше: в 1.4 раза для a и в 2.5 раза для b.
#   * Оставшаяся погрешность — в основном шум: он случайный, размазан по всем
#     коэффициентам понемногу, и никакой базис его хорошо не сожмёт.
