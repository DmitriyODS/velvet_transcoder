"""
Тема 5. Вейвлет-преобразование изображений. Задание 2. Устранение шума.

Идея вейвлет-очистки от шума:
  1) раскладываем зашумлённое изображение по вейвлетам;
  2) у настоящих деталей изображения (края, контуры) коэффициенты большие,
     а шум «размазан» по всем коэффициентам понемногу. Поэтому обнуляем или
     уменьшаем все коэффициенты деталей меньше порога T;
  3) восстанавливаем изображение.

Два вида порога:
  * жёсткий (hard): если |c| < T, то c = 0, иначе c не меняем;
  * мягкий (soft): если |c| < T, то c = 0, иначе |c| уменьшаем на T
    (c → sign(c)·(|c| - T)). Нет резких скачков, картинка глаже.

Два способа выбрать порог T:
  * VisuShrink (универсальный): T = σ·√(2·ln N), где σ — уровень шума,
    N — число пикселей. Порог один для всего изображения и довольно
    большой, поэтому часто «замыливает» картинку.
  * BayesShrink (байесовский): для каждой полосы коэффициентов свой порог
    T = σ² / σ_сигнала. Где сигнала много (σ_сигнала большое), порог ниже,
    и детали сохраняются.

Качество оценивается по PSNR (пиковое отношение сигнал/шум) в децибелах:
    PSNR = 10·log10(MAX² / MSE),
где MSE — средний квадрат отличия от оригинала, MAX = 1 (максимальная яркость).
Чем больше PSNR, тем ближе картинка к оригиналу.
+3 дБ означает, что средняя квадратичная ошибка уменьшилась вдвое.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

np.random.seed(4321)

img = img_as_float(data.camera())    # то же изображение, что в задании 1


def psnr(x):
    return peak_signal_noise_ratio(img, x, data_range=1)


def show(images, titles, filename):
    fig, ax = plt.subplots(1, len(images), figsize=(6*len(images), 6))
    for a, im, tt in zip(np.atleast_1d(ax), images, titles):
        a.imshow(im, cmap='gray', vmin=0, vmax=1)
        a.set_title(tt)
        a.axis('off')
    fig.tight_layout()
    fig.savefig(RESULTS / filename)
    plt.show()


# ---------------------------------------------------------------------------
# a) Белый (гауссов) шум со стандартным отклонением σ = 0.5
# ---------------------------------------------------------------------------
# К каждому пикселю добавляется случайное число из нормального распределения
# с дисперсией σ² = 0.25. Для яркостей в диапазоне [0; 1] это очень сильный шум.
# random_noise обрезает результат до [0; 1].
sigma = 0.5
imgn = random_noise(img, var=sigma**2, rng=4321)

# Оценка уровня шума по самому зашумлённому изображению.
# В реальной задаче оригинала у нас нет, поэтому оцениваем по тому, что есть.
# estimate_sigma смотрит на самые мелкие диагональные детали, где почти нет
# ничего, кроме шума. Оценка меньше 0.5, потому что обрезка до [0; 1]
# «съела» часть шума.
sigma_est = estimate_sigma(imgn, average_sigmas=True)
print(f"Оценка уровня шума sigma_est = {sigma_est:.4f}")

# b) PSNR до очистки
psnr_noisy = psnr(imgn)
print(f"b) PSNR зашумлённого изображения: {psnr_noisy:.2f} дБ")

# c) исходное и зашумлённое изображения
show([img, imgn], ['исходное', f'с шумом σ = {sigma}, PSNR = {psnr_noisy:.2f} дБ'],
     'task2c_noisy.png')

table = {}


def denoise(label, wavelet, mode, method, sigma_value=None):
    """Очищает изображение, печатает и запоминает PSNR."""
    # wavelet_levels не задаём — берётся максимально возможное число уровней.
    # rescale_sigma=True — после очистки поправить масштаб σ (рекомендуется).
    out = denoise_wavelet(imgn, wavelet=wavelet, mode=mode, method=method,
                          sigma=sigma_value, rescale_sigma=True)
    value = psnr(out)
    table[label] = value
    print(f"{label}: PSNR = {value:.2f} дБ")
    return out


# ---------------------------------------------------------------------------
# d–f) Байесовский метод: мягкий и жёсткий порог
# ---------------------------------------------------------------------------
bs_soft = denoise("d) haar, soft, BayesShrink", 'haar', 'soft', 'BayesShrink')
bs_hard = denoise("e) haar, hard, BayesShrink", 'haar', 'hard', 'BayesShrink')
best_bayes = 'soft' if table["d) haar, soft, BayesShrink"] >= table["e) haar, hard, BayesShrink"] else 'hard'
print(f"f) для BayesShrink лучше порог: {best_bayes}\n")
show([bs_soft, bs_hard], ['Bayes, soft', 'Bayes, hard'], 'task2_def_bayes.png')

# ---------------------------------------------------------------------------
# g–i) Универсальный метод: мягкий и жёсткий порог
# ---------------------------------------------------------------------------
# Если sigma не передать, denoise_wavelet сам оценит уровень шума.
vs_soft = denoise("g) haar, soft, VisuShrink", 'haar', 'soft', 'VisuShrink')
vs_hard = denoise("h) haar, hard, VisuShrink", 'haar', 'hard', 'VisuShrink')
best_visu = 'soft' if table["g) haar, soft, VisuShrink"] >= table["h) haar, hard, VisuShrink"] else 'hard'
print(f"i) для VisuShrink лучше порог: {best_visu}\n")
show([vs_soft, vs_hard], ['Visu, soft', 'Visu, hard'], 'task2_ghi_visu.png')

# ---------------------------------------------------------------------------
# j–l) Универсальный метод, лучший порог, разные sigma
# ---------------------------------------------------------------------------
# Порог VisuShrink пропорционален sigma. sigma_est/3 даёт порог в 3 раза меньше:
# удаляется меньше шума, но и меньше деталей изображения.
vs_full = denoise(f"j) haar, {best_visu}, Visu, sigma = sigma_est",
                  'haar', best_visu, 'VisuShrink', sigma_est)
vs_third = denoise(f"k) haar, {best_visu}, Visu, sigma = sigma_est/3",
                   'haar', best_visu, 'VisuShrink', sigma_est/3)
best_sigma = 'sigma_est' if (table[f"j) haar, {best_visu}, Visu, sigma = sigma_est"]
                             >= table[f"k) haar, {best_visu}, Visu, sigma = sigma_est/3"]) else 'sigma_est/3'
print(f"l) лучше использовать sigma = {best_sigma}\n")
show([vs_full, vs_third], ['Visu, sigma_est', 'Visu, sigma_est/3'], 'task2_jkl_sigma.png')

# ---------------------------------------------------------------------------
# m) Лучший метод и параметры
# ---------------------------------------------------------------------------
# Сравниваем лучший результат Bayes и лучший результат Visu (с выбранной sigma)
candidates = {
    ('BayesShrink', best_bayes, None): max(table["d) haar, soft, BayesShrink"],
                                           table["e) haar, hard, BayesShrink"]),
    ('VisuShrink', best_visu, best_sigma): max(table[f"j) haar, {best_visu}, Visu, sigma = sigma_est"],
                                               table[f"k) haar, {best_visu}, Visu, sigma = sigma_est/3"]),
}
best_method, best_mode, best_sig_name = max(candidates, key=candidates.get)
best_sig_value = {None: None, 'sigma_est': sigma_est, 'sigma_est/3': sigma_est/3}[best_sig_name]
print(f"m) лучший метод: {best_method}, порог {best_mode}"
      + (f", sigma = {best_sig_name}" if best_sig_name else "") + "\n")

# ---------------------------------------------------------------------------
# n, o) Тот же метод с вейвлетами db4, sym4, bior4.4 и итоговая таблица
# ---------------------------------------------------------------------------
final = {'haar': candidates[(best_method, best_mode, best_sig_name)]}
outs = []
for wname in ['db4', 'sym4', 'bior4.4']:
    out = denoise(f"n) {wname}, {best_mode}, {best_method}",
                  wname, best_mode, best_method, best_sig_value)
    final[wname] = table[f"n) {wname}, {best_mode}, {best_method}"]
    outs.append(out)
show(outs, [f'{w}: {final[w]:.2f} дБ' for w in ['db4', 'sym4', 'bior4.4']],
     'task2n_wavelets.png')

print("\no) Таблица PSNR (дБ):")
print(f"   {'без очистки':>12}: {psnr_noisy:.2f}")
for w, v in final.items():
    print(f"   {w:>12}: {v:.2f}")
print(f"   лучший вейвлет для устранения шума: {max(final, key=final.get)}")

# Как понимать результат (PSNR, дБ): шум 9.18 → после очистки 16.9–18.0.
#   * BayesShrink: soft (17.73) и hard (17.70) почти одинаковы. Порог у Bayes
#     небольшой, поэтому мягкое «поджатие» коэффициентов почти не мешает.
#   * VisuShrink: hard (17.18) лучше soft (16.36). Универсальный порог очень
#     большой, и мягкий порог дополнительно уменьшает ВСЕ оставшиеся
#     коэффициенты на эту большую величину, поэтому картинка «замыливается».
#   * sigma_est лучше, чем sigma_est/3 (17.19 против 12.27): шум очень сильный
#     (σ = 0.5), и порог, уменьшенный втрое, оставляет большую часть шума.
#     sigma_est/3 имеет смысл только при слабом шуме.
#   * Лучший метод — BayesShrink с мягким порогом: он подбирает свой порог
#     для каждой полосы коэффициентов.
#   * По вейвлетам: db4 ≈ sym4 (17.95) > haar (17.73) > bior4.4 (16.89).
#     Гладкие ортогональные вейвлеты дают меньше «квадратных» артефактов, чем
#     haar. bior4.4 неортогональный: шум после разложения по нему уже не
#     «белый», оценка шума по порогам искажается (об этом и предупреждает
#     skimage), поэтому результат хуже.
