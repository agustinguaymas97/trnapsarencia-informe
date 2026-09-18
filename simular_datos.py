import pandas as pd
import numpy as np

# Parámetros del experimento "ideal pero realista"
I_0 = 1600.0        # Lux iniciales reales
alpha_real = 0.04   # Atenuación verdadera del folio (4% por capa)
ruido_std = 15.0    # Desviación estándar del ruido en lux (para que no sea perfecto)

# 1. Generar los folios (n de 0 a 16)
n_capas = np.arange(17)

# 2. Calcular la curva teórica perfecta: I = I_0 * e^(-alpha * n)
I_teorica = I_0 * np.exp(-alpha_real * n_capas)

# 3. Inyectar ruido aleatorio (Gaussiano)
np.random.seed(42) # Semilla fija para que te dé siempre los mismos datos
ruido = np.random.normal(loc=0.0, scale=ruido_std, size=len(n_capas))
I_medida = I_teorica + ruido

# Redondear a 1 decimal (como suelen mostrar los celulares)
I_medida = np.round(I_medida, 1)

# 4. Crear la tabla y mostrarla
datos = {
    'Folios (n)': n_capas,
    'Lux Teóricos': np.round(I_teorica, 1),
    'Lux Medidos (Con Ruido)': I_medida
}

df = pd.DataFrame(datos)

print("=== DATOS SIMULADOS (REALISTAS) ===")
print(df.to_string(index=False))

# 5. Guardar como Excel (simulando la salida de la app, asumiendo 1 meseta por folio)
# Para simplificar, ponemos el número de folio como "Time" solo para que tus scripts lo lean
df_export = pd.DataFrame({
    'Time (s)': n_capas * 20, # Simulando que mediste cada 20 segundos
    'Illuminance (lx)': I_medida
})
df_export.to_excel("corrida_ideal.xlsx", index=False)
print("\n¡Archivo 'corrida_ideal.xlsx' guardado!")
