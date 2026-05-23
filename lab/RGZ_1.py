import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

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

def OFDM_symbol(complex_array):
    symbol = np.zeros(K, dtype=complex)
    symbol[pilot_carriers] = pilot_value
    symbol[data_carriers] = complex_array
    return symbol

def add_awgn(signal, snr_dB):
    snr_linear = 10**(snr_dB / 10)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def channel_estimation_LS(rx_symbols, pilot_carriers, pilot_value):
    Y_p = rx_symbols[pilot_carriers]
    
    H_LS = Y_p / pilot_value
    
    return H_LS        
    
def interpolate_channel(H_p, pilot_carriers, K):
    H_interp = np.zeros(K, dtype=complex)
    all_carriers = np.arange(K)
    
    H_interp = np.interp(all_carriers, pilot_carriers, H_p)
    
    return H_interp

K = 64
Pilot_step = 4
P = K // Pilot_step
CP_len = K // 4
pilot_value = 1 + 0j
N_sym = 1

carriers = np.arange(K)
pilot_carriers = carriers[::K//P]
data_carriers = np.delete(carriers, pilot_carriers)

data_bits_per_symbol = len(data_carriers) * 2

H_matr = sp.loadmat("H16_umi_nlos_2.mat")
H16 = H_matr["H16_16_Umi_NLOS"]

matrix_size = 8
N_bits = 10**4
rank = 4

H = H16[0:matrix_size, 0:matrix_size, 0:64, 1]

H_norm = (H / np.linalg.norm(H))

H_est_all = np.empty((matrix_size, matrix_size), dtype=object)

for i in range(matrix_size):
    for j in range(matrix_size):
        OFDM_time_symbols = np.array([], dtype=complex)
        
        data = np.random.randint(0, 2, data_bits_per_symbol)
        QPSK_array = [mapper_QPSK(data[i], data[i + 1]) for i in range(0, len(data), 2)]
        
        OFDM_data = OFDM_symbol(QPSK_array)
        
        OFDM_data = OFDM_symbol(QPSK_array)
        OFDM_time = np.fft.ifft(OFDM_data)
        
        cyclic_prefix = OFDM_time[-CP_len:]
        OFDM_time = np.hstack([cyclic_prefix, OFDM_time])
        
        OFDM_time_symbols = np.hstack([OFDM_time_symbols, OFDM_time])
        
        h = np.fft.ifft(H_norm[i, j, :])
        
        y = np.convolve(h, OFDM_time_symbols)
        
        OFDM_time_channel = add_awgn(y, snr_dB=20)
        
        OFDM_rx = OFDM_time_channel[CP_len:]
        OFDM_rx = OFDM_rx[:K]

        OFDM_freq = np.fft.fft(OFDM_rx)

        H_LS = channel_estimation_LS(OFDM_freq[:K], pilot_carriers, pilot_value)
        H_est = interpolate_channel(H_LS, pilot_carriers, K)
        
        H_est_all[i, j] = H_est
        
H_mimo = np.zeros((matrix_size, matrix_size), dtype=complex)      
for i in range(matrix_size):
    for j in range(matrix_size):
        h_ij = H_est_all[i, j]
        H_mimo[i, j] = h_ij[0]

U, E, V = np.linalg.svd(H_mimo)

indices = range(1, len(E) + 1)

plt.figure(figsize=(10, 6))
plt.bar(indices, E)

bits = np.random.randint(0, 2, N_bits)

QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                       for i in range(0, len(bits), 2)])

N_symbols = N_bits // 2
N_vectors = N_symbols // rank

x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)

V = V[:rank, :]
U = U[:, :rank]
E = E[:rank]

z_all = []
for i in range(N_vectors):
    x = x_vectors[i, :]
    
    x_precoded = V.conj().T @ x
    
    y = H_mimo @ x_precoded
    
    y_noise = add_awgn(y, 10)
    
    z = U.conj().T @ y_noise
    z = z / E
    
    z_all.append(z)

z_all = np.array(z_all)

plt.figure(figsize=(10, 6))
plt.scatter(z_all[:, 0].real, z_all[:, 0].imag)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Сигнальная диаграмма на выходе схемы комбинирования для выбранного слоя')
plt.grid()
plt.show()

# SER calc for SVD

SNR_dB_range = np.arange(0, 17, 1)
SER_results_SVD = []

for snr_dB in SNR_dB_range:
    print(f"SNR = {snr_dB} дБ")
    total_symbol_errors = 0
    total_symbols = 0
    
    for _ in range(100):
        bits = np.random.randint(0, 2, N_bits)

        QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                               for i in range(0, len(bits), 2)])

        rank = 4
        N_symbols = N_bits // 2
        N_vectors = N_symbols // rank

        x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)
        
        z_all = []
        for i in range(N_vectors):
            x = x_vectors[i, :]
            
            x_precoded = V.conj().T @ x
            
            y = H_mimo @ x_precoded
            
            y_noise = add_awgn(y, snr_dB)
            
            z = U.conj().T @ y_noise
            z = z / E
            
            z_all.append(z)

        z_all = np.array(z_all)
        
        z_flat = z_all.flatten()
        
        for i in range(len(z_flat)):
            tx_symbol = QPSK_array[i]
            rx_symbol = z_flat[i]
            
            if np.sign(tx_symbol.real) != np.sign(rx_symbol.real) or \
               np.sign(tx_symbol.imag) != np.sign(rx_symbol.imag):
                total_symbol_errors += 1
        
        total_symbols += len(z_flat)
    
    SER = total_symbol_errors / total_symbols
    SER_results_SVD.append(SER)
    print(f"SER (SVD) = {SER:.6f}\n")
        
# ZF

bits = np.random.randint(0, 2, N_bits)

QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                       for i in range(0, len(bits), 2)])

N_symbols = N_bits // 2
N_vectors = N_symbols // rank

x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)

H_mimo_red = H_mimo[:, :rank]

w_r = np.linalg.inv(H_mimo_red.conj().T @ H_mimo_red) @ H_mimo_red.conj().T

z_all = []
for i in range(N_vectors):
    x = x_vectors[i, :]
    
    r = H_mimo_red @ x
    
    r_noise = add_awgn(r, 10)
    
    z = w_r @ r_noise
    
    z_all.append(z)

z_all = np.array(z_all)

plt.figure(figsize=(10, 6))
plt.scatter(z_all[:, 0].real, z_all[:, 0].imag)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Сигнальная диаграмма на выходе схемы комбинирования для выбранного слоя (ZF)')
plt.grid()
plt.show()

# SER calc for ZF

SNR_dB_range = np.arange(0, 17, 1)
SER_results_ZF = []

for snr_dB in SNR_dB_range:
    print(f"SNR = {snr_dB} дБ")
    total_symbol_errors = 0
    total_symbols = 0
    
    for _ in range(100):
        bits = np.random.randint(0, 2, N_bits)

        QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                               for i in range(0, len(bits), 2)])

        rank = 4
        N_symbols = N_bits // 2
        N_vectors = N_symbols // rank

        x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)
        
        w_r = np.linalg.inv(H_mimo_red.conj().T @ H_mimo_red) @ H_mimo_red.conj().T

        z_all = []
        for i in range(N_vectors):
            x = x_vectors[i, :]
            
            r = H_mimo_red @ x
            
            r_noise = add_awgn(r, snr_dB)
            
            z = w_r @ r_noise
            
            z_all.append(z)

        z_all = np.array(z_all)
        
        z_flat = z_all.flatten()
        
        for i in range(len(z_flat)):
            tx_symbol = QPSK_array[i]
            rx_symbol = z_flat[i]
            
            if np.sign(tx_symbol.real) != np.sign(rx_symbol.real) or \
               np.sign(tx_symbol.imag) != np.sign(rx_symbol.imag):
                total_symbol_errors += 1
        
        total_symbols += len(z_flat)
    
    SER = total_symbol_errors / total_symbols
    SER_results_ZF.append(SER)
    print(f"SER (ZF) = {SER:.6f}\n")

# MMSE

bits = np.random.randint(0, 2, N_bits)

QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                       for i in range(0, len(bits), 2)])

N_symbols = N_bits // 2
N_vectors = N_symbols // rank

x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)

H_mimo_red = H_mimo[:, :rank]

snr_dB = 10
signal_power = 1.0
snr_linear = 10**(snr_dB / 10)
H_power = np.mean(np.abs(H_mimo_red)**2)
received_signal_power = signal_power * H_power
sigma_n_squared = received_signal_power / snr_linear

w_r_MMSE = (np.linalg.inv(H_mimo_red.conj().T @ H_mimo_red + sigma_n_squared 
                          * np.eye(rank)) @ H_mimo_red.conj().T)

z_all = []
for i in range(N_vectors):
    x = x_vectors[i, :]
    
    r = H_mimo_red @ x
    
    r_noise = add_awgn(r, snr_dB)

    z = w_r_MMSE @ r_noise
    
    z_all.append(z)

z_all = np.array(z_all)

plt.figure(figsize=(10, 6))
plt.scatter(z_all[:, 0].real, z_all[:, 0].imag)
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Сигнальная диаграмма на выходе схемы комбинирования для выбранного слоя (MMSE)')
plt.grid()
plt.show()

SNR_dB_range = np.arange(0, 17, 1)
SER_results_MMSE = []

for snr_dB in SNR_dB_range:
    print(f"SNR = {snr_dB} дБ")
    total_symbol_errors = 0
    total_symbols = 0
    
    for _ in range(100):
        bits = np.random.randint(0, 2, N_bits)

        QPSK_array = np.array([mapper_QPSK(bits[i], bits[i + 1]) 
                               for i in range(0, len(bits), 2)])

        rank = 4
        N_symbols = N_bits // 2
        N_vectors = N_symbols // rank

        x_vectors = QPSK_array[:N_vectors * rank].reshape(N_vectors, rank)
        
        signal_power = 1.0
        snr_linear = 10**(snr_dB / 10)
        H_power = np.mean(np.abs(H_mimo_red)**2)
        received_signal_power = signal_power * H_power
        sigma_n_squared = received_signal_power / snr_linear

        w_r_MMSE = (np.linalg.inv(H_mimo_red.conj().T @ H_mimo_red + sigma_n_squared 
                                  * np.eye(rank)) @ H_mimo_red.conj().T)

        z_all = []
        for i in range(N_vectors):
            x = x_vectors[i, :]
            
            r = H_mimo_red @ x
            
            r_noise = add_awgn(r, snr_dB)

            z = w_r_MMSE @ r_noise
            
            z_all.append(z)

        z_all = np.array(z_all)
        
        z_flat = z_all.flatten()
        
        for i in range(len(z_flat)):
            tx_symbol = QPSK_array[i]
            rx_symbol = z_flat[i]
            
            if np.sign(tx_symbol.real) != np.sign(rx_symbol.real) or \
               np.sign(tx_symbol.imag) != np.sign(rx_symbol.imag):
                total_symbol_errors += 1
        
        total_symbols += len(z_flat)
    
    SER = total_symbol_errors / total_symbols
    SER_results_MMSE.append(SER)
    print(f"SER (MMSE) = {SER:.6f}\n")
    
plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dB_range, SER_results_SVD, 'bo-', label='SVD')
plt.semilogy(SNR_dB_range, SER_results_ZF, 'rs-', label='ZF')
plt.semilogy(SNR_dB_range, SER_results_MMSE, 'go-', label='MMSE')
plt.xlabel('SNR (дБ)')
plt.ylabel('SER')
plt.title('Зависимость SER от SNR', fontsize=14)
plt.legend()
plt.xlim([0, 17])
plt.grid()
plt.tight_layout()
plt.show()
