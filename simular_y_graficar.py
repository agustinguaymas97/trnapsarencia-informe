import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# ==========================================
# 1. GENERACIÓN DE LOS 5 EXCEL SIMULADOS
# ==========================================
np.random.seed(42) # Semilla para que el "caos" sea controlado y reproducible

n_capas = np.arange(17) # 0 hasta 16 folios
duracion_escalon = 20   # 20 segundos por escalón
dt = 1.0                # 1 muestra por segundo

print("Generando archivos de simulación...")
for run_idx in range(1, 6):
    tiempo_total = []
    lux_total = []
    t_actual = 0.0

    # Fondo inicial (20 segundos)
    t_fondo = np.arange(0, 20, dt)
    l_fondo = np.random.normal(10.0, 0.8, len(t_fondo))
    tiempo_total.extend(t_fondo)
    lux_total.extend(l_fondo)
    t_actual = 20.0

    # Pequeña variación entre corridas para que no sean idénticas
    alpha_real = 0.035 + np.random.normal(0, 0.0015)
    I0_real = 1600.0 + np.random.normal(0, 10.0)

    # Escalones (0 a 16) - Acá el I0 (n=0) coincide perfecto con la teórica
    for n in n_capas:
        t_esc = np.arange(t_actual, t_actual + duracion_escalon, dt)
        I_teorica = I0_real * np.exp(-alpha_real * n)
        
        # Ruido realista del 1.5% (simula pulso de la mano y micro-vibraciones)
        l_esc = np.random.normal(I_teorica, I_teorica * 0.015, len(t_esc))
        
        tiempo_total.extend(t_esc)
        lux_total.extend(l_esc)
        t_actual += duracion_escalon

    df_run = pd.DataFrame({
        'Time (s)': tiempo_total,
        'Illuminance (lx)': lux_total
    })
    df_run.to_excel(f"run{run_idx}.xlsx", index=False)

print("¡Archivos run1.xlsx a run5.xlsx generados con éxito!\n")

# ==========================================
# 2. ANÁLISIS Y GRAFICACIÓN AUTOMÁTICA
# ==========================================
UMBRAL_ESTABILIDAD = 3.0      
TIEMPO_MINIMO_MESETA = 12.0   

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Simulación Experimental: Linealización de la Ley de Beer-Lambert", fontsize=13)
ax.set_xlabel("Número de folios ($n$)", fontsize=12)
ax.set_ylabel("$\\ln(I_{medido} - I_{fondo})$", fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)

alphas = []

for run_idx in range(1, 6):
    archivo = f"run{run_idx}.xlsx"
    df = pd.read_excel(archivo)
    tiempo, lux = df['Time (s)'].values, df['Illuminance (lx)'].values

    # Detección de mesetas
    delta_t = np.diff(tiempo)
    delta_lux = np.diff(lux)
    derivada = np.abs(delta_lux / delta_t)
    derivada = np.insert(derivada, 0, 0)
    
    media_ruido = np.median(derivada)
    mascara_estable = derivada < (media_ruido * UMBRAL_ESTABILIDAD)
    
    mesetas = []
    meseta_actual = []
    for i, estable in enumerate(mascara_estable):
        if estable:
            meseta_actual.append(i)
        else:
            if len(meseta_actual) > 0 and (tiempo[meseta_actual[-1]] - tiempo[meseta_actual[0]]) >= TIEMPO_MINIMO_MESETA:
                mesetas.append(meseta_actual)
            meseta_actual = []
            
    if len(meseta_actual) > 0 and (tiempo[meseta_actual[-1]] - tiempo[meseta_actual[0]]) >= TIEMPO_MINIMO_MESETA:
        mesetas.append(meseta_actual)

    if len(mesetas) < 3:
        continue

    medias = np.array([np.mean(lux[idx]) for idx in mesetas])
    fondo = medias[0]
    I_n = medias[1:] - fondo
    validos = I_n > 0
    I_n = I_n[validos]
    n_capas_val = np.arange(len(I_n))

    y = np.log(I_n)
    resultado = linregress(n_capas_val, y)
    alpha = -resultado.slope
    alphas.append(alpha)

    # Graficar puntos y recta de ajuste
    ax.scatter(n_capas_val, y, label=f"Run {run_idx}", s=20)
    ax.plot(n_capas_val, resultado.intercept + resultado.slope * n_capas_val, linestyle='--', alpha=0.7)

# Estadística final
if alphas:
    alpha_medio = np.mean(alphas)
    error_estandar = np.std(alphas, ddof=1) / np.sqrt(len(alphas))

    print("="*50)
    print(f"RESULTADO FINAL SIMULADO")
    print("="*50)
    print(f"Alpha promedio : {alpha_medio:.4f}")
    print(f"Incertidumbre  : ± {error_estandar:.4f}")
    print("="*50)
    
    texto_resultado = f"$\\alpha_{{sim}} = {alpha_medio:.4f} \\pm {error_estandar:.4f}$\n($N=5$ corridas simuladas)"
    ax.text(0.95, 0.95, texto_resultado, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend()
plt.tight_layout()
plt.savefig("grafico_simulado.png", dpi=300)
print("\n¡Gráfico guardado con éxito como 'grafico_simulado.png'!")
