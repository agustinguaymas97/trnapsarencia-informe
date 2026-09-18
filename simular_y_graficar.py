import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# ==========================================
# 1. GENERACIÓN DE LOS 5 EXCEL SIMULADOS (MÁS CAÓTICOS)
# ==========================================
np.random.seed(123) # Semilla nueva para un desorden diferente

n_capas = np.arange(17) 
duracion_escalon = 20   
dt = 1.0                

print("Generando archivos de simulación desordenados...")
for run_idx in range(1, 6):
    tiempo_total = []
    lux_total = []
    t_actual = 0.0

    t_fondo = np.arange(0, 20, dt)
    l_fondo = np.random.normal(15.0, 3.0, len(t_fondo)) # Más ruido de fondo
    tiempo_total.extend(t_fondo)
    lux_total.extend(l_fondo)
    t_actual = 20.0

    # Mucha más dispersión entre pasadas (caja movida, lámpara que calienta)
    alpha_base = 0.040 + np.random.normal(0, 0.008) 
    I0_real = 1600.0 + np.random.normal(0, 80.0)

    for n in n_capas:
        t_esc = np.arange(t_actual, t_actual + duracion_escalon, dt)
        
        # Simulamos que cada folio no es perfecto (algunos más opacos, otros más transparentes)
        imperfeccion_folio = np.random.normal(1.0, 0.035) 
        
        I_teorica = I0_real * np.exp(-alpha_base * n) * imperfeccion_folio
        
        # Sensor captando reflexiones raras y ruido (4% de ruido en vez de 1.5%)
        l_esc = np.random.normal(I_teorica, I_teorica * 0.04, len(t_esc))
        
        tiempo_total.extend(t_esc)
        lux_total.extend(l_esc)
        t_actual += duracion_escalon

    df_run = pd.DataFrame({'Time (s)': tiempo_total, 'Illuminance (lx)': lux_total})
    df_run.to_excel(f"run{run_idx}.xlsx", index=False)

print("¡Archivos caóticos generados!\n")

# ==========================================
# 2. ANÁLISIS Y GRAFICACIÓN AUTOMÁTICA
# ==========================================
UMBRAL_ESTABILIDAD = 5.0      # Subimos la tolerancia porque los datos son más ruidosos
TIEMPO_MINIMO_MESETA = 10.0   

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Linealización de la Ley de Beer-Lambert (Datos Realistas)", fontsize=13)
ax.set_xlabel("Número de folios ($n$)", fontsize=12)
ax.set_ylabel("$\\ln(I_{medido} - I_{fondo})$", fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)

alphas = []

for run_idx in range(1, 6):
    archivo = f"run{run_idx}.xlsx"
    df = pd.read_excel(archivo)
    tiempo, lux = df['Time (s)'].values, df['Illuminance (lx)'].values

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

    # Gráfico
    ax.scatter(n_capas_val, y, label=f"Run {run_idx}", s=20)
    ax.plot(n_capas_val, resultado.intercept + resultado.slope * n_capas_val, linestyle='--', alpha=0.7)

if alphas:
    alpha_medio = np.mean(alphas)
    error_estandar = np.std(alphas, ddof=1) / np.sqrt(len(alphas))

    print("="*50)
    print(f"Alpha promedio : {alpha_medio:.4f} ± {error_estandar:.4f}")
    print("="*50)
    
    texto_resultado = f"$\\alpha_{{medido}} = {alpha_medio:.4f} \\pm {error_estandar:.4f}$\n($N=5$ corridas)"
    ax.text(0.95, 0.95, texto_resultado, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend()
plt.tight_layout()
plt.savefig("grafico_simulado.png", dpi=300)
print("\n¡Gráfico guardado! (Fijate ahora cómo se dispersan los puntos)")
