"""
Тема 5. Вейвлет-преобразование изображений. Задание 1. Сжатие изображений.

Изображение — это таблица (матрица) яркостей пикселей. Двумерное DWT
выполняется как одномерное, только дважды: сначала по строкам, потом по столбцам.
После одного уровня получаются 4 «картинки» вдвое меньшего размера:
    cA — аппроксимация: уменьшенная и сглаженная копия изображения;
    cH — горизонтальные детали (перепады яркости сверху вниз →
         видны горизонтальные края);
    cV — вертикальные детали (перепады слева направо → вертикальные края);
    cD — диагональные детали.
При многоуровневом разложении cA раскладывается снова, и так далее.

Сжатие: большинство коэффициентов деталей близки к нулю (на ровных участках
картинки перепадов нет). Оставляем заданную долю самых больших по модулю
коэффициентов, остальные обнуляем и восстанавливаем картинку.
«Коэффициент сжатия 0.5» здесь понимается как «оставить 50% коэффициентов»
(второй вариант формулировки из пункта i).

Качество оцениваем нормой разности ||исходное - восстановленное||:
это корень из суммы квадратов ошибок во всех пикселях. Чем меньше, тем лучше.

Изображение: стандартное тестовое 'cameraman' из skimage (512x512, ч/б).
Яркости переведены в диапазон [0; 1].
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import data, img_as_float

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

KEEP = 0.5                            # оставляем 50% коэффициентов
WAVELETS = ['haar', 'db4', 'sym4', 'bior4.4']
MULTI_LEVEL = 3                       # число уровней для многоуровневого разложения

# a) Загрузка изображения
img = img_as_float(data.camera())     # матрица 512x512, числа от 0 (чёрный) до 1 (белый)
print("Размер изображения:", img.shape)


def keep_largest(arr, keep):
    """Обнуляет все элементы, кроме доли keep самых больших по модулю."""
    threshold = np.quantile(np.abs(arr), 1 - keep)
    return np.where(np.abs(arr) >= threshold, arr, 0)


def compress(image, wavelet, level):
    """Разложение → оставить KEEP коэффициентов → восстановление.

    coeffs_to_array складывает все коэффициенты всех уровней в одну большую
    матрицу (как мозаику), чтобы общий порог можно было применить сразу ко всем.
    array_to_coeffs разбирает мозаику обратно на уровни.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)
    arr_c = keep_largest(arr, KEEP)
    coeffs_c = pywt.array_to_coeffs(arr_c, slices, output_format='wavedec2')
    rec = pywt.waverec2(coeffs_c, wavelet)
    # Для длинных фильтров размер может выйти чуть больше — обрезаем
    rec = rec[:image.shape[0], :image.shape[1]]
    return coeffs, arr, rec


def show_one_level(coeffs, wavelet):
    """Картинки cA, cH, cV, cD одноуровневого разложения."""
    cA, (cH, cV, cD) = coeffs
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    for a, c, name in zip(ax.flat, [cA, cH, cV, cD], ['cA', 'cH', 'cV', 'cD']):
        a.imshow(c, cmap='gray')
        a.set_title(name)
        a.axis('off')
    fig.suptitle(f"b) одноуровневое разложение, {wavelet}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"task1_decomp_{wavelet}_level1.png")
    plt.show()


def show_multi_level(arr, wavelet, level):
    """Все коэффициенты многоуровневого разложения одной мозаикой."""
    plt.figure(figsize=(8, 8))
    # Коэффициенты деталей малы по сравнению с cA; модуль и степень 0.3
    # делают их заметнее (иначе почти всё поле было бы чёрным).
    plt.imshow(np.abs(arr)**0.3, cmap='gray')
    plt.title(f"h) все коэффициенты, {wavelet}, {level} уровня")
    plt.axis('off')
    plt.savefig(RESULTS / f"task1_decomp_{wavelet}_level{level}.png")
    plt.show()


def show_compare(rec, wavelet, level, err):
    """Исходное и сжатое изображения рядом."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('исходное')
    ax[1].imshow(rec, cmap='gray')
    ax[1].set_title(f"сжатое ({int(KEEP*100)}% коэф.), {wavelet}, уровней: {level}\n"
                    f"||разность|| = {err:.4f}")
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(RESULTS / f"task1_compare_{wavelet}_level{level}.png")
    plt.show()


results = {}
for level in [1, MULTI_LEVEL]:
    print(f"\n=== Разложение на {level} {'уровень' if level == 1 else 'уровня'}")
    for wname in WAVELETS:
        # b, h) разложение;  c, i) сжатие и восстановление
        coeffs, arr, rec = compress(img, wname, level)
        # e, k) норма разности
        err = np.linalg.norm(img - rec)
        results[(wname, level)] = err
        print(f"   {wname:>8}: ||исходное - сжатое|| = {err:.4f}")

        if level == 1:
            show_one_level(coeffs, wname)
        else:
            show_multi_level(arr, wname, level)
        # d, j) исходное и сжатое изображения на одном рисунке
        show_compare(rec, wname, level, err)

# g, m) Выводы
for level in [1, MULTI_LEVEL]:
    best = min(WAVELETS, key=lambda w: results[(w, level)])
    print(f"\nЛучший вейвлет при {level} уровн{'е' if level == 1 else 'ях'}: {best}")

# Как понимать результат (норма ошибки: 1 уровень / 3 уровня):
#     haar 1.55 / 0.94,  db4 1.38 / 0.80,  sym4 1.38 / 0.80,  bior4.4 1.35 / 0.77
#   * При одном уровне cA — это четверть всех коэффициентов, и все они большие,
#     поэтому попадают в оставленные 50%. Вторая половина бюджета уходит на
#     самые сильные детали (края объектов). Выбрасываются мелкие детали — почти
#     незаметная «рябь», поэтому на глаз сжатое изображение почти не отличается.
#   * При трёх уровнях аппроксимация занимает всего 1/64 коэффициентов,
#     и на детали остаётся больше бюджета, поэтому ошибка почти вдвое меньше.
#   * Гладкие вейвлеты описывают плавные переходы яркости лучше, чем
#     «ступенчатый» haar. Лучший в обоих случаях — bior4.4: у него симметричные
#     фильтры и 4 нулевых момента, поэтому края объектов и плавные области
#     описываются малым числом коэффициентов.
