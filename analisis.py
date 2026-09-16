import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIGURACIÓN OPTIMIZADA PARA TUS DATOS ---
UMBRAL_ESTABILIDAD = 3.0      # Más permisivo para no perder los 16 folios
TIEMPO_MINIMO_MESETA = 12.0   # Pedimos 12s estables dentro de tus 20s
COLUMNA_TIEMPO = 'Time (s)'
COLUMNA_LUX = 'Illuminance (lx)'

def procesar_corrida(ruta_archivo, ax_plot):
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

    # 2. Extraer Medias y Desviaciones de cada meseta
    medias = np.array([np.mean(lux[idx]) for idx in mesetas])
    desv = np.array([np.std(lux[idx]) for idx in mesetas])
    
    fondo = medias[0]
    I_n = medias[1:]
    sigma_I = desv[1:]
    
    # 3. Filtrar datos válidos
    validos = I_n > fondo
    I_n = I_n[validos] - fondo
    sigma_I = sigma_I[validos]
    n_capas = np.arange(len(I_n))

    # 4. Logaritmo y propagación de error
    y = np.log(I_n)
    sigma_y = sigma_I / I_n # Propagación del error relativo
    
    # 5. Ajuste lineal ponderado y estadística
    pesos = 1.0 / (sigma_y**2)
    coefs, cov = np.polyfit(n_capas, y, 1, w=pesos, cov=True)
    pendiente, ordenada = coefs[0], coefs[1]
    
    alpha = -pendiente
    
    # Cálculo de R^2
    y_ajuste = pendiente * n_capas + ordenada
    ss_res = np.sum((y - y_ajuste)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Cálculo de Chi^2 reducido
    chi2 = np.sum(((y - y_ajuste) / sigma_y)**2)
    grados_libertad = len(n_capas) - 2
    chi2_red = chi2 / grados_libertad if grados_libertad > 0 else 0

    # 6. Graficar
    nombre_run = os.path.basename(ruta_archivo)
    ax_plot.scatter(n_capas, y, label=f"{nombre_run}", s=20)
    ax_plot.plot(n_capas, y_ajuste, linestyle='--', alpha=0.7)

    return {'Corrida': nombre_run, 'Alpha': alpha, 'R^2': r2, 'Chi^2_red': chi2_red, 'Puntos_Usados': len(n_capas)}

def analizar_todo():
    archivos = sorted(glob.glob("*.xlsx"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Linealización de la Ley de Beer-Lambert", fontsize=14)
    ax.set_xlabel("Número de folios ($n$)", fontsize=12)
    ax.set_ylabel("$\ln(I_{medido} - I_{fondo})$", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    resultados = []
    for archivo in archivos:
        res = procesar_corrida(archivo, ax)
        if res is not None:
            resultados.append(res)

    if resultados:
        # Generar tabla ordenada en la terminal
        df_res = pd.DataFrame(resultados)
        print("\n" + "="*65)
        print("=== TABLA DE RESULTADOS POR CORRIDA (ANÁLISIS DE ERRORES) ===")
        print("="*65)
        print(df_res.to_string(index=False, float_format="%.4f"))
        print("="*65)

        alphas = df_res['Alpha'].values
        alpha_medio = np.mean(alphas)
        error_estandar = np.std(alphas, ddof=1) / np.sqrt(len(alphas))

        print(f"\nRESULTADO FINAL: Alpha_promedio = {alpha_medio:.4f} ± {error_estandar:.4f}")
        
        texto_resultado = f"$\\alpha_{{final}} = {alpha_medio:.4f} \\pm {error_estandar:.4f}$\n($N={len(alphas)}$ corridas)"
        ax.text(0.95, 0.95, texto_resultado, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend()
    plt.tight_layout()
    plt.savefig("grafico_beer_lambert.png", dpi=300)
    print("\n¡Gráfico guardado con éxito! Revisá la imagen y la tabla superior.")

if __name__ == '__main__':
    analizar_todo()
