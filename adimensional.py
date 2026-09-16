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

def procesar_adimensional(ruta_archivo, ax_plot):
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
    # MODELO ADIMENSIONAL FORZADO AL ORIGEN
    # ==========================================
    # Solo ajustamos para n >= 1 (ya que en n=0 da exactamente 0 por definición)
    n_capas_ajuste = n_capas[1:]
    I_n_ajuste = I_n[1:]
    sigma_I_ajuste = sigma_I[1:]
    
    y = np.log(I_n_ajuste / I_0)
    
    # Propagación de error: suma de errores relativos al cuadrado
    sigma_y = np.sqrt((sigma_I_ajuste / I_n_ajuste)**2 + (sigma_I0 / I_0)**2)
    pesos = 1.0 / (sigma_y**2)
    
    # Regresión forzada al origen (y = m*x)
    m = np.sum(pesos * n_capas_ajuste * y) / np.sum(pesos * n_capas_ajuste**2)
    alpha = -m
    
    y_ajuste = -alpha * n_capas_ajuste
    
    # Estadísticas
    ss_res = np.sum((y - y_ajuste)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Graficar
    ax_plot.scatter(n_capas_ajuste, y, label=nombre_run, s=20)
    # Trazamos la línea desde el origen (0,0)
    n_plot = np.insert(n_capas_ajuste, 0, 0)
    y_plot = -alpha * n_plot
    ax_plot.plot(n_plot, y_plot, linestyle='--', alpha=0.7)

    return {'Corrida': nombre_run, 'Alpha_Adim': alpha, 'R^2': r2}

def analizar_adimensional():
    archivos = sorted(glob.glob("*.xlsx"))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.set_title("Ley de Beer-Lambert (Modelo Adimensional)", fontsize=14)
    ax.set_xlabel("Número de folios ($n$)", fontsize=12)
    ax.set_ylabel("$\\ln(I_n / I_0)$", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    resultados = []
    for archivo in archivos:
        res = procesar_adimensional(archivo, ax)
        if res is not None:
            resultados.append(res)

    if resultados:
        df_res = pd.DataFrame(resultados)
        print("\n" + "="*50)
        print("=== RESULTADOS MODELO ADIMENSIONAL ===")
        print("="*50)
        print(df_res.to_string(index=False, float_format="%.4f"))
        print("="*50)

        alpha_medio = df_res['Alpha_Adim'].mean()
        err = df_res['Alpha_Adim'].std(ddof=1) / np.sqrt(len(df_res))

        print(f"\nAlpha Adimensional Promedio: {alpha_medio:.4f} ± {err:.4f}")
        
        texto_resultado = f"$\\alpha_{{adim}} = {alpha_medio:.4f} \\pm {err:.4f}$\n($N={len(df_res)}$ corridas)"
        ax.text(0.95, 0.05, texto_resultado, transform=ax.transAxes, 
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend()
    plt.tight_layout()
    plt.savefig("grafico_adimensional.png", dpi=300)
    print("\n¡Gráfico guardado como 'grafico_adimensional.png'!")

if __name__ == '__main__':
    analizar_adimensional()
