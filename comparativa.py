import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIGURACIÓN ---
UMBRAL_ESTABILIDAD = 3.0      
TIEMPO_MINIMO_MESETA = 12.0   
COLUMNA_TIEMPO = 'Time (s)'
COLUMNA_LUX = 'Illuminance (lx)'

def procesar_ambos_modelos(ruta_archivo, ax1, ax2):
    df = pd.read_excel(ruta_archivo)
    
    if COLUMNA_TIEMPO not in df.columns or COLUMNA_LUX not in df.columns:
        tiempo_col, lux_col = df.columns[0], df.columns[1]
    else:
        tiempo_col, lux_col = COLUMNA_TIEMPO, COLUMNA_LUX

    tiempo, lux = df[tiempo_col].values, df[lux_col].values

    # 1. Detección de escalones
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
        return None

    # 2. Extraer Medias y Desviaciones
    medias = np.array([np.mean(lux[idx]) for idx in mesetas])
    desv = np.array([np.std(lux[idx]) for idx in mesetas])
    
    fondo = medias[0]
    I_n = medias[1:] - fondo
    sigma_I = desv[1:]
    
    validos = I_n > 0
    I_n = I_n[validos]
    sigma_I = sigma_I[validos]
    n_capas = np.arange(len(I_n))

    # El I0 es el punto n=0
    I_0 = I_n[0]
    sigma_I0 = sigma_I[0]

    nombre_run = os.path.basename(ruta_archivo)

    # ==========================================
    # MODELO 1: ORDENADA LIBRE (ln(I_n) vs n)
    # ==========================================
    y1 = np.log(I_n)
    sigma_y1 = sigma_I / I_n # Propagación relativa simple
    pesos1 = 1.0 / (sigma_y1**2)
    
    coefs1 = np.polyfit(n_capas, y1, 1, w=pesos1)
    alpha1 = -coefs1[0]
    ordenada1 = coefs1[1]
    
    y1_ajuste = -alpha1 * n_capas + ordenada1
    ss_res1 = np.sum((y1 - y1_ajuste)**2)
    ss_tot1 = np.sum((y1 - np.mean(y1))**2)
    r2_1 = 1 - (ss_res1 / ss_tot1)

    ax1.scatter(n_capas, y1, label=nombre_run, s=15)
    ax1.plot(n_capas, y1_ajuste, linestyle='--', alpha=0.7)

    # ==========================================
    # MODELO 2: ADIMENSIONAL FORZADO (ln(I_n/I_0) vs n)
    # ==========================================
    # Para n=0, ln(I0/I0) = 0. Ajustamos solo los puntos n >= 1
    n_capas2 = n_capas[1:]
    I_n2 = I_n[1:]
    sigma_I2 = sigma_I[1:]
    
    y2 = np.log(I_n2 / I_0)
    
    # Propagación de error CRÍTICA: Se suma el error relativo de In y el de I0
    sigma_y2 = np.sqrt((sigma_I2 / I_n2)**2 + (sigma_I0 / I_0)**2)
    pesos2 = 1.0 / (sigma_y2**2)
    
    # Ajuste lineal forzado al origen (y = m*x sin 'b')
    # Fórmula de mínimos cuadrados ponderados para pendiente pura: m = sum(w*x*y) / sum(w*x^2)
    m = np.sum(pesos2 * n_capas2 * y2) / np.sum(pesos2 * n_capas2**2)
    alpha2 = -m
    
    y2_ajuste = -alpha2 * n_capas2
    ss_res2 = np.sum((y2 - y2_ajuste)**2)
    ss_tot2 = np.sum((y2 - np.mean(y2))**2)
    r2_2 = 1 - (ss_res2 / ss_tot2)

    ax2.scatter(n_capas2, y2, label=nombre_run, s=15)
    # Agregamos el punto (0,0) artificialmente para la visualización de la recta
    ax2.plot(np.insert(n_capas2, 0, 0), -alpha2 * np.insert(n_capas2, 0, 0), linestyle='--', alpha=0.7)

    return {
        'Corrida': nombre_run, 
        'Alpha_Libre': alpha1, 'R2_Libre': r2_1,
        'Alpha_Adim': alpha2, 'R2_Adim': r2_2
    }

def analizar_comparativa():
    archivos = sorted(glob.glob("*.xlsx"))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title("Modelo 1: Ordenada Libre\n$\\ln(I_n) = -\\alpha n + \\ln(I_0)$", fontsize=12)
    ax1.set_xlabel("Número de folios ($n$)")
    ax1.set_ylabel("$\\ln(I_{medido})$")
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_title("Modelo 2: Adimensional Forzado al Origen\n$\\ln(I_n / I_0) = -\\alpha n$", fontsize=12)
    ax2.set_xlabel("Número de folios ($n$)")
    ax2.set_ylabel("$\\ln(I_n / I_0)$")
    ax2.grid(True, linestyle=':', alpha=0.6)

    resultados = []
    for archivo in archivos:
        res = procesar_ambos_modelos(archivo, ax1, ax2)
        if res is not None:
            resultados.append(res)

    if resultados:
        df_res = pd.DataFrame(resultados)
        print("\n" + "="*80)
        print("=== COMPARATIVA DE MODELOS ===")
        print("="*80)
        print(df_res.to_string(index=False, float_format="%.4f"))
        print("="*80)

        # Promedios finales
        alpha1_medio = df_res['Alpha_Libre'].mean()
        err1 = df_res['Alpha_Libre'].std(ddof=1) / np.sqrt(len(df_res))
        
        alpha2_medio = df_res['Alpha_Adim'].mean()
        err2 = df_res['Alpha_Adim'].std(ddof=1) / np.sqrt(len(df_res))

        print(f"\nRESULTADO MODELO 1 (Libre)       : Alpha = {alpha1_medio:.4f} ± {err1:.4f}")
        print(f"RESULTADO MODELO 2 (Adimensional): Alpha = {alpha2_medio:.4f} ± {err2:.4f}")

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig("comparativa_modelos.png", dpi=300)
    print("\n¡Gráfico guardado como 'comparativa_modelos.png'!")

if __name__ == '__main__':
    analizar_comparativa()
