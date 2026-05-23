import numpy as np
import matplotlib.pyplot as plt

def mapper_QPSK(bit_1, bit_2):
    I = 0
    Q = 0
    if bit_1 == 0:
        if bit_2 == 0:
            I = 0.707
            Q = 0.707
        elif bit_2 == 1:
            I = 0.707
            Q = -0.707
    elif bit_1 == 1:
        if bit_2 == 0:
            I = -0.707
            Q = 0.707
        elif bit_2 == 1:
            I = -0.707
            Q = -0.707
    return complex(I, Q)

def demapper_QPSK(QPSK_array):
    bit_sequence = [0] * (2 * len(QPSK_array))
    for i in range(len(QPSK_array)):
        QPSK_real = QPSK_array[i].real
        QPSK_imag = QPSK_array[i].imag
        
        if QPSK_real >= 0 and QPSK_imag >= 0:
            bit_sequence[2*i] = 0
            bit_sequence[2*i + 1] = 0
        elif QPSK_real >= 0 and QPSK_imag < 0:
            bit_sequence[2*i] = 0
            bit_sequence[2*i + 1] = 1
        elif QPSK_real < 0 and QPSK_imag >= 0:
            bit_sequence[2*i] = 1
            bit_sequence[2*i + 1] = 0
        else: 
            bit_sequence[2*i] = 1
            bit_sequence[2*i + 1] = 1

    return bit_sequence

def add_awgn(signal, snr_dB):
    snr_linear = 10**(snr_dB / 10)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def steering_vector(N, d, lam, theta):
    k = 2 * np.pi / lam
    n = np.arange(N)
    phase = -1j * k * d * np.sin(theta) * n[None, :]
    return np.exp(phase)

theta_deg = 35
theta = np.deg2rad(theta_deg)

c = 3e8
f = 3.5e9
lam = c / f
d = lam / 2

# 1

a = steering_vector(2, d, lam, theta).reshape(-1, 1)

# 2

np.random.seed(42)
bits = np.random.randint(0, 2, 10**4)

QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                       for i in range(0, len(bits), 2)])

# 3

g0_db = -10
g0_lin = 10**(g0_db * 0.1)
h = g0_lin * a

r = h @ QPSK_array[None, :]

# 4

r = add_awgn(r, 10)

# 5

a4_err = steering_vector(8, d, lam, 40).reshape(-1, 1)
w_opt = np.conj(a).T

# 6

z = w_opt @ r

plt.figure(figsize=(10, 6))
plt.scatter(r[0].real, r[0].imag)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Сигнальная диаграмма на одном антенном элементе')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(z.real, z.imag)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Сигнальная диаграмма на выходе схемы сложения')
plt.grid()
plt.show()

# 7

snr_range = np.arange(0, 13, 1)
ser_single = np.zeros(len(snr_range))
ser_array_4 = np.zeros(len(snr_range))
ser_array_8 = np.zeros(len(snr_range))

for _ in range(100):
    bits = np.random.randint(0, 2, 10**4)
    bits_reshaped = bits.reshape(-1, 2)
    QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                           for i in range(0, len(bits), 2)])
    
    for idx, snr_dB in enumerate(snr_range):
        a4 = steering_vector(2, d, lam, theta).reshape(-1, 1)
        a8 = steering_vector(4, d, lam, theta).reshape(-1, 1)
        
        h_4 = g0_lin * a4
        h_8 = g0_lin * a8
        
        r_single = h_4[0, 0] * QPSK_array
        r_single = add_awgn(r_single, snr_dB)
        
        r_4 = h_4 @ QPSK_array[None, :]
        r_4 = add_awgn(r_4, snr_dB)
        
        r_8 = h_8 @ QPSK_array[None, :]
        r_8 = add_awgn(r_8, snr_dB)
        
        w_opt_4 = np.conj(a4).T
        z_4 = w_opt_4 @ r_4
        z_4 = z_4.flatten()
        
        w_opt_8 = np.conj(a8).T
        z_8 = w_opt_8 @ r_8
        z_8 = z_8.flatten()
        
        dec_single = np.array(demapper_QPSK(r_single)).reshape(-1, 2)
        dec_4 = np.array(demapper_QPSK(z_4)).reshape(-1, 2)
        dec_8 = np.array(demapper_QPSK(z_8)).reshape(-1, 2)
        
        symbol_errors_single = np.sum(np.any(dec_single != bits_reshaped, axis=1))
        symbol_errors_4 = np.sum(np.any(dec_4 != bits_reshaped, axis=1))
        symbol_errors_8 = np.sum(np.any(dec_8 != bits_reshaped, axis=1))
        
        num_symbols = len(QPSK_array)
        ser_single[idx] += symbol_errors_single / num_symbols
        ser_array_4[idx] += symbol_errors_4 / num_symbols
        ser_array_8[idx] += symbol_errors_8 / num_symbols

ser_single /= 100
ser_array_4 /= 100
ser_array_8 /= 100
        
plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, ser_single, 'o-', label='Одна антенна')
plt.semilogy(snr_range, ser_array_4, 's-', label='АР 4 элемента')
plt.semilogy(snr_range, ser_array_8, '^-', label='АР 8 элементов')
plt.xlabel('SNR, дБ')
plt.ylabel('SER')
plt.title('Зависимость SER от SNR')
plt.grid()
plt.legend()
plt.xlim([0, 12])
plt.show()
        

