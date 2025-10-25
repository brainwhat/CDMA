from typing import List, Dict, Optional
import math
import random


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def build_hadamard_matrix(order: int) -> List[List[int]]:
    """Вернуть матрицу Уолша (Адамара) с элементами +1/-1.

    Используется конструкция Сильвестра:
      H1 = [1]
      H(2n) = [ H(n)  H(n) ; H(n)  -H(n) ]

    Порядок должен быть степенью двойки (1, 2, 4, 8, ...).
    """
    if not _is_power_of_two(order):
        raise ValueError("Hadamard order must be a power of two (e.g., 1,2,4,8,...)")

    matrix: List[List[int]] = [[1]]
    size = 1
    while size < order:
        top: List[List[int]] = []
        bottom: List[List[int]] = []

        for row in matrix:
            top.append(row + row)
        for row in matrix:
            bottom.append(row + [-x for x in row])

        matrix = top + bottom
        size *= 2

    return matrix

def assign_station_codes(h8: List[List[int]]) -> Dict[str, List[int]]:
    """Назначить четыре различных кода Уолша (длина 8) станциям A–D из заданной H8.

    Избегаем нулевую строку (все +1, DC) и берём строки 1..4 — они нулесредние.
    """
    return {
        "A": h8[1],
        "B": h8[2],
        "C": h8[3],
        "D": h8[4],
    }

def ascii_to_bits(text: str) -> List[int]:
    """Преобразовать текст ASCII в список битов 0/1 (8 бит на символ, MSB→LSB)."""
    bits: List[int] = []
    for ch in text:
        code_point = ord(ch)
        # Ограничить к 8-битному диапазону ASCII (0..255); при выходе взять младшие 8 бит.
        code_point &= 0xFF
        for bit_index in range(7, -1, -1):
            bit = (code_point >> bit_index) & 1
            bits.append(bit)
    return bits

def bits_to_chips(bits: List[int]) -> List[int]:
    """Отобразить биты в BPSK-чипы: 1 → +1, 0 → −1."""
    return [1 if b == 1 else -1 for b in bits]

def spread_signal(data_chips: List[int], walsh_code: List[int]) -> List[int]:
    """Растянуть (spread) чипы данных кодом Уолша (длина 8).

    Для каждого чипа s \\in {+1,−1} выдаём последовательность s * walsh_code.
    Длина результата = len(data_chips) * len(walsh_code).
    """
    if not walsh_code:
        return []
    spread: List[int] = []
    for chip in data_chips:
        for c in walsh_code:
            spread.append(chip * c)
    return spread

def sum_signals(signals: List[List[int]]) -> List[int]:
    """Суммировать несколько spread‑сигналов поэлементно 

    Все входные сигналы должны иметь одинаковую длину. Пустой вход → пустой результат.
    """
    if not signals:
        return []
    length = len(signals[0])
    for s in signals[1:]:
        if len(s) != length:
            raise ValueError("All signals must have the same length for summation")
    return [sum(values) for values in zip(*signals)]

def sigma_from_snr_db(snr_db: float) -> float:
    """Вычислить СКО шума (sigma) по SNR на чип в дБ при мощности сигнала = 1.

    SNR = Ps/Pn при Ps=1 ⇒ sigma = sqrt(Pn) = 1 / sqrt(10^(SNR_dB/10)).
    """
    return 1.0 / math.sqrt(10.0 ** (snr_db / 10.0))

def add_awgn_noise(signal: List[int], sigma: float, seed: Optional[int] = None) -> List[float]:
    """Добавить AWGN N(0, sigma^2) к дискретному сигналу и вернуть список float."""
    rng = random.Random(seed) if seed is not None else random
    return [float(x) + rng.gauss(0.0, sigma) for x in signal]

def despread_and_detect(channel: List[float], walsh_code: List[int]) -> List[int]:
    """Деспрединг канала по коду Уолша и принятие решения по знаку корреляции.

    Возвращает список детектированных чипов из {+1, −1}. Длина канала должна
    быть кратна длине кода.
    """
    if not walsh_code:
        return []
    n = len(walsh_code)
    if len(channel) % n != 0:
        raise ValueError("Channel length must be a multiple of the code length")
    detected: List[int] = []
    for i in range(0, len(channel), n):
        block = channel[i:i + n]
        r = sum(x * c for x, c in zip(block, walsh_code))
        detected.append(1 if r >= 0 else -1)
    return detected

def chips_to_bits(chips: List[int]) -> List[int]:
    """Преобразовать чипы {+1, −1} в биты {1, 0} с порогом 0."""
    return [1 if c >= 0 else 0 for c in chips]

def bits_to_ascii(bits: List[int]) -> str:
    """Преобразовать список битов (MSB→LSB, длина кратна 8) в строку ASCII."""
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8")
    chars: List[str] = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        value = 0
        for b in byte_bits:
            value = (value << 1) | (1 if b else 0)
        chars.append(chr(value))
    return "".join(chars)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDMA simulation with Walsh codes")
    parser.add_argument("--show-matrix", action="store_true", help="Print H8 matrix")
    parser.add_argument("--show-codes", action="store_true", help="Print assigned station codes")
    parser.add_argument("--debug", action="store_true", help="Print intermediate signals")
    parser.add_argument("--snr-db", type=float, default=None, help="Add AWGN with given SNR per chip in dB")
    parser.add_argument("--noise-seed", type=int, default=None, help="Random seed for noise generator")
    args = parser.parse_args()

    # Сгенерировать матрицу Уолша и назначить коды станциям
    h8 = build_hadamard_matrix(8)
    codes = assign_station_codes(h8)

    if args.show_matrix:
        print("H8 (Walsh-Hadamard) matrix:")
        for row in h8:
            print(" ".join(f"{v:2d}" for v in row))

    if args.show_codes:
        print("\nAssigned station codes (rows of H8):")
        for station, code in codes.items():
            try:
                idx = next(i for i, row in enumerate(h8) if row == code)
            except StopIteration:
                idx = -1
            print(f"{station} [H8 row {idx}]:", " ".join(f"{v:2d}" for v in code))

    # Построить spread‑сигналы для всех станций (их слова)
    station_words = {"A": "GOD", "B": "CAT", "C": "HAM", "D": "SUN"}
    signals: List[List[int]] = []
    per_station = {}
    for st, word in station_words.items():
        st_bits = ascii_to_bits(word)
        st_chips = bits_to_chips(st_bits)
        st_spread = spread_signal(st_chips, codes[st])
        per_station[st] = {
            "word": word,
            "bits": st_bits,
            "chips": st_chips,
            "spread": st_spread,
        }
        signals.append(st_spread)

    if args.debug:
        print("\nPer-station intermediate data:")
        for st in ["A", "B", "C", "D"]:
            info = per_station[st]
            bits_str = "".join(str(b) for b in info["bits"])
            chips_str = " ".join(f"{c:+d}" for c in info["chips"])
            spread_preview = " ".join(f"{x:+d}" for x in info["spread"][:32])
            print(f"{st} word: {info['word']}")
            print(f"{st} bits: {bits_str}")
            print(f"{st} chips: {chips_str}")
            print(f"{st} spread (first 32): {spread_preview}")

    channel = sum_signals(signals)

    # Опционально добавить шум
    channel_for_rx: List[float]
    if args.snr_db is not None:
        sigma = sigma_from_snr_db(args.snr_db)
        channel_for_rx = add_awgn_noise(channel, sigma, seed=args.noise_seed)
        if args.debug:
            print(f"\nAdded AWGN: SNR={args.snr_db:.2f} dB, sigma={sigma:.4f}")
            print("Noisy channel (first 32):")
            print(" ".join(f"{v:+.2f}" for v in channel_for_rx[:32]))
    else:
        channel_for_rx = [float(v) for v in channel]

    if args.debug:
        print("\nChannel summed signal (first 32):")
        print(" ".join(f"{v:+d}" for v in channel[:32]))

    # Восстановить слова всех станций
    print("Recovered words per station:")
    for st in ["A", "B", "C", "D"]:
        detected = despread_and_detect(channel_for_rx, codes[st])
        rec_bits = chips_to_bits(detected)
        rec_text = bits_to_ascii(rec_bits)
        print(f"{st}: {rec_text}")
        if args.debug:
            det_preview = " ".join(f"{c:+d}" for c in detected[:16])
            print(f"{st} detected chips (first 16): {det_preview}")


