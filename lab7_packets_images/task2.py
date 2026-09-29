"""
Тема 7. Вейвлет-пакеты и изображения. Задание 2.
Двумерный вейвлет-пакет на 2 уровня и выбор наилучшего дерева.

Двумерный вейвлет-пакет: на каждом шаге узел делится на 4 части:
    a — аппроксимация (уменьшенная сглаженная картинка),
    h — горизонтальные детали, v — вертикальные, d — диагональные.
В отличие от обычного 2D DWT (тема 5), дальше делятся ВСЕ 4 части,
а не только a. На 2 уровнях получаем 4 + 16 = 20 узлов:
    уровень 1: a, h, v, d
    уровень 2: aa, ah, av, ad, ha, hh, ..., dd

Наилучшее дерево выбираем так же, как в задании 1: по энтропии Шеннона
-Σ p ln p, где p = c² / E — доля энергии изображения в коэффициенте. Для каждого узла 1-го уровня сравниваем его стоимость
с суммой стоимостей его 4 детей и оставляем более дешёвый вариант.
Затем так же сравниваем корень (всю картинку) с лучшим разбиением.

Для сжатия в выбранном базисе оставляем 10% наибольших коэффициентов
и восстанавливаем картинку. Для сравнения то же делается
на полном дереве уровня 1 и уровня 2.

Вейвлет: sym4 (гладкий, почти симметричный — хорошо подходит для фотографий).
Изображение: 'cameraman' из skimage, 512x512.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

WAVELET = 'sym4'
KEEP = 0.10
PARTS = ['a', 'h', 'v', 'd']

img = img_as_float(data.camera())
H, W = img.shape

wp = pywt.WaveletPacket2D(data=img, wavelet=WAVELET, mode='symmetric', maxlevel=2)


ENERGY = np.sum(img**2)   # энергия всего изображения


def entropy(c):
    """Энтропия Шеннона: -Σ p ln p, p = c² / (энергия изображения).

    p — доля энергии, приходящаяся на коэффициент. Меньше энтропия →
    энергия собрана в меньшем числе коэффициентов → лучше для сжатия.
    """
    p = c[c != 0]**2 / ENERGY
    return -np.sum(p * np.log(p))


# ---------------------------------------------------------------------------
# Компоненты изображения на 1-м и 2-м уровнях
# ---------------------------------------------------------------------------
def show_node(ax, node):
    d = node.data
    # У аппроксимации значения большие и положительные — показываем как есть;
    # у деталей — модуль в степени 0.5, чтобы слабые перепады стали видны.
    ax.imshow(d if set(node.path) == {'a'} else np.abs(d)**0.5, cmap='gray')
    ax.set_title(node.path, fontsize=9)
    ax.axis('off')


fig, ax = plt.subplots(2, 2, figsize=(8, 8))
for a, node in zip(ax.flat, wp.get_level(1, 'natural')):
    show_node(a, node)
fig.suptitle(f"Уровень 1 ({WAVELET})")
fig.tight_layout()
fig.savefig(RESULTS / "task2_level1.png")
plt.show()

# natural-порядок уровня 2: aa, ah, av, ad, ha, ... — строка = родитель
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for a, node in zip(ax.flat, wp.get_level(2, 'natural')):
    show_node(a, node)
fig.suptitle(f"Уровень 2 ({WAVELET}): строка — родительский узел a / h / v / d")
fig.tight_layout()
fig.savefig(RESULTS / "task2_level2.png")
plt.show()

# ---------------------------------------------------------------------------
# Выбор наилучшего дерева
# ---------------------------------------------------------------------------
cost = {n.path: entropy(n.data) for n in wp.get_level(1) + wp.get_level(2)}
basis = []
print("Сравнение стоимости (энтропии) узла и его 4 детей:")
for p in PARTS:
    children = sum(cost[p + q] for q in PARTS)
    choice = 'делим' if children < cost[p] else 'не делим'
    print(f"   {p}: сам {cost[p]:.5f}, дети {children:.5f} → {choice}")
    basis += [p + q for q in PARTS] if children < cost[p] else [p]
total_basis = sum(cost[p] for p in basis)
root_cost = entropy(img)
print(f"   корень (без разложения): {root_cost:.5f}, лучшее разбиение: {total_basis:.5f}")
print("Наилучшее дерево (листья):", basis)


# ---------------------------------------------------------------------------
# Сжатие и восстановление в выбранном базисе
# ---------------------------------------------------------------------------
def compress(paths):
    """Оставляет KEEP наибольших коэффициентов в узлах paths и восстанавливает."""
    all_c = np.concatenate([wp[p].data.ravel() for p in paths])
    thr = np.quantile(np.abs(all_c), 1 - KEEP)
    leaves = {}
    kept = 0
    for p in paths:
        c = np.where(np.abs(wp[p].data) >= thr, wp[p].data, 0)
        kept += np.count_nonzero(c)
        leaves[p] = c
    return rebuild('', leaves), kept, len(all_c)


def rebuild(path, leaves):
    """Восстанавливает узел path из листьев снизу вверх.

    Если узел — лист выбранного дерева, берём его (сжатые) коэффициенты.
    Иначе восстанавливаем 4 детей и делаем обратное 2D-преобразование:
    idwt2(a, (h, v, d)). Результат обрезаем до исходного размера узла:
    из-за продления краёв у sym4 он может выйти на 1 пиксель больше.
    (Стандартный reconstruct из PyWavelets на разноуровневом дереве
    падает именно из-за этого расхождения размеров.)
    """
    if path in leaves:
        return leaves[path]
    a, h, v, d = (rebuild(path + q, leaves) for q in PARTS)
    rec = pywt.idwt2((a, (h, v, d)), WAVELET, mode='symmetric')
    rows, cols = (wp[path].data.shape if path else img.shape)
    return rec[:rows, :cols]


variants = {
    'полное дерево, уровень 1': [n.path for n in wp.get_level(1, 'natural')],
    'полное дерево, уровень 2': [n.path for n in wp.get_level(2, 'natural')],
    'наилучшее дерево': basis,
}
recs = {}
print(f"\nСжатие: оставляем {int(KEEP*100)}% коэффициентов")
for name, paths in variants.items():
    rec, kept, total = compress(paths)
    err = np.linalg.norm(img - rec)
    psnr = peak_signal_noise_ratio(img, np.clip(rec, 0, 1), data_range=1)
    recs[name] = (rec, err, psnr)
    print(f"   {name:>26}: {kept} из {total} коэф., ||ошибка|| = {err:.3f}, PSNR = {psnr:.2f} дБ")

fig, ax = plt.subplots(1, 4, figsize=(22, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('исходное')
for a, (name, (rec, err, psnr)) in zip(ax[1:], recs.items()):
    a.imshow(np.clip(rec, 0, 1), cmap='gray')
    a.set_title(f"{name}\nPSNR = {psnr:.2f} дБ")
for a in ax:
    a.axis('off')
fig.tight_layout()
fig.savefig(RESULTS / "task2_reconstruction.png")
plt.show()

# Как понимать результат:
#   * Узел 'a' выгодно делить: в нём вся «картинка», и после деления её
#     энергия собирается в маленьком 'aa'.
#   * Узлы деталей h, v, d делить невыгодно: их энтропия после деления растёт.
#     У фотографии они состоят из почти нулевых значений с редкими всплесками
#     на контурах, и дальнейшее деление только «размазывает» эти всплески.
#   * Итоговое дерево разноуровневое: {aa, ah, av, ad, h, v, d}. По сути это
#     обычное 2D DWT на 2 уровня (как в теме 5), и это разумно: для обычных
#     фотографий классическое вейвлет-разложение уже почти оптимально.
#   * Сжатие до 10%:
#       - полный уровень 1: PSNR ≈ 10 дБ, очень плохо. Узел 'a' — это 25%
#         коэффициентов, и все они большие; бюджета 10% не хватает даже на него;
#       - полный уровень 2 (32.6 дБ) и наилучшее дерево (32.4 дБ) почти
#         одинаковы. Наилучшее дерево хранит меньше коэффициентов в сумме
#         (272 тыс. против 283 тыс.: меньше «лишних» краевых значений),
#         но энтропия — это мера компактности, а не прямо ошибка восстановления,
#         поэтому по PSNR оно не обязано выигрывать.
