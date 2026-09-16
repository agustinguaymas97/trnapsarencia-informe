import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import glob
import os

# --- CONFIGURACIÓN ---
UMBRAL_ESTABILIDAD = 1.5      # Sensibilidad para detectar saltos (ruido)
TIEMPO_MINIMO_MESETA = 10.0   # Segundos mínimos que debe durar un escalón para ser válido
COLUMNA_TIEMPO = 'Time (s)'       # Cambiar si Phyphox lo exportó distinto
COLUMNA_LUX = 'Illuminance (lx)'  # Cambiar si Phyphox lo exportó distinto

def procesar_corrida(ruta_archivo, ax_plot):
    print(f"\n--- Procesando: {ruta_archivo} ---")
    
    # CAMBIO ACÁ: Usamos read_excel para los .xlsx
    df = pd.read_excel(ruta_archivo)
    
    # Manejo de nombres de columnas si cambian un poco
    if COLUMNA_TIEMPO not in df.columns or COLUMNA_LUX not in df.columns:
        tiempo_col = df.columns[0] # Asume que la primera es tiempo
        lux_col = df.columns[1]    # Asume que la segunda es lux
    else:
        tiempo_col, lux_col = COLUMNA_TIEMPO, COLUMNA_LUX

    tiempo = df[tiempo_col].values
    lux = df[lux_col].values

    # 1. Detección de escalones (Mesetas)
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
            if len(meseta_actual) > 0:
                t_inicio, t_fin = tiempo[meseta_actual[0]], tiempo[meseta_actual[-1]]
                if (t_fin - t_inicio) >= TIEMPO_MINIMO_MESETA:
                    mesetas.append(meseta_actual)
            meseta_actual = []
            
    # Guardar la última si cortaste justo
    if len(meseta_actual) > 0 and (tiempo[meseta_actual[-1]] - tiempo[meseta_actual[0]]) >= TIEMPO_MINIMO_MESETA:
        mesetas.append(meseta_actual)

    print(f"Se detectaron {len(mesetas)} escalones estables (Se esperaban 18: 1 Fondo + 1 I0 + 16 Folios).")

    # 2. Extracción de promedios
    promedios = [np.mean(lux[indices]) for indices in mesetas]
    
    if len(promedios) < 3:
        print("Error: No se detectaron suficientes escalones. Revisar los datos.")
        return None

    fondo = promedios[0]
    intensidades = np.array(promedios[1:]) # El resto (I0 en adelante)
    
    # 3. Limpieza y Linealización
    intensidades_reales = intensidades - fondo
    intensidades_reales = intensidades_reales[intensidades_reales > 0] # Evitar log(0) o negativos
    
    n_capas = np.arange(len(intensidades_reales))
    y = np.log(intensidades_reales)

    # 4. Regresión Lineal
    resultado = linregress(n_capas, y)
    alpha = -resultado.slope
    r_cuadrado = resultado.rvalue**2

    print(f"Alpha medido: {alpha:.4f} | R^2: {r_cuadrado:.4f}")

    # 5. Graficar en el plot general
    nombre_run = os.path.basename(ruta_archivo)
    ax_plot.scatter(n_capas, y, label=f"{nombre_run} (Datos)", s=20)
    ax_plot.plot(n_capas, resultado.intercept + resultado.slope * n_capas, linestyle='--', alpha=0.7)

    return alpha

def analizar_todo():
    # CAMBIO ACÁ: Buscamos archivos .xlsx
    archivos = sorted(glob.glob("*.xlsx"))
    
    if not archivos:
        print("No se encontraron archivos .xlsx en esta carpeta.")
        return

    # Preparar el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Linealización de la Ley de Beer-Lambert (5 Corridas)", fontsize=14)
    ax.set_xlabel("Número de folios ($n$)", fontsize=12)
    ax.set_ylabel("$\ln(I_{medido} - I_{fondo})$", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    alphas = []
    for archivo in archivos:
        alpha = procesar_corrida(archivo, ax)
        if alpha is not None:
            alphas.append(alpha)

    # Estadística final
    if alphas:
        alpha_medio = np.mean(alphas)
        alpha_std = np.std(alphas, ddof=1) # Desviación estándar muestral
        error_estandar = alpha_std / np.sqrt(len(alphas)) # Incertidumbre del promedio

        print("\n" + "="*40)
        print("=== RESULTADO FINAL DEL EXPERIMENTO ===")
        print("="*40)
        print(f"Coeficiente de Atenuación Promedio (Alpha) : {alpha_medio:.4f}")
        print(f"Incertidumbre (Error Estándar de la Media) : ± {error_estandar:.4f}")
        print(f"Dispersión entre corridas (Desv. Estándar) : {alpha_std:.4f}")
        print("="*40)

        # Agregar el resultado final como caja de texto en el gráfico
        texto_resultado = f"$\\alpha_{{final}} = {alpha_medio:.4f} \\pm {error_estandar:.4f}$ por capa\n($N={len(alphas)}$ corridas)"
        ax.text(0.95, 0.95, texto_resultado, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend()
    plt.tight_layout()
    plt.savefig("grafico_beer_lambert.png", dpi=300)
    print("\n¡Gráfico guardado como 'grafico_beer_lambert.png'!")

if __name__ == '__main__':
    analizar_todo()
