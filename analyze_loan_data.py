import pandas as pd
import numpy as np

# Nombre del archivo
file_path = 'loan_ejercicio_clean.xlsx'
output_file_path = 'loan_data_clasificado_final.xlsx'

# Cargar el archivo
try:
    df = pd.read_excel(file_path)
    total_clients = len(df)
    print(f"Total de clientes en la base: {total_clients}")

    # 1. Cálculo de la variable de Recuperación
    # Recuperación = total_rec_prncp / funded_amnt
    df['recuperacion_pct'] = (df['total_rec_prncp'] / df['funded_amnt']) * 100

    # 2. Definición de las Variables Secundarias (4 condiciones)
    df['sec_cond_1'] = df['dti'] <= 22
    df['sec_cond_2'] = df['grade'] == 'A'
    # Manejo de NaNs en delinq_2yrs: Rellenamos con un valor que no cumpla la condición (ej. -1) para que NaN sea False
    df['sec_cond_3'] = (df['delinq_2yrs'].fillna(-1) == 0)
    df['sec_cond_4'] = df['recuperacion_pct'] >= 70

    # 3. Conteo de condiciones secundarias cumplidas
    secondary_conditions = ['sec_cond_1', 'sec_cond_2', 'sec_cond_3', 'sec_cond_4']
    df['num_sec_cond_cumplidas'] = df[secondary_conditions].sum(axis=1)
    is_secondary_ok = df['num_sec_cond_cumplidas'] >= 3

    # 4. Definición de la Condición Principal Inicial (C_P_30)
    # mths_since_last_delinq >= 30 o la celda este vacia (NaN)
    is_principal_ok_30 = (df['mths_since_last_delinq'] >= 30) | (df['mths_since_last_delinq'].isna())

    # 5. Definición de la Condición Principal Estricta (C_P_36) para BMB
    # mths_since_last_delinq >= 36 o la celda este vacia (NaN)
    is_principal_ok_36 = (df['mths_since_last_delinq'] >= 36) | (df['mths_since_last_delinq'].isna())

    # 6. Clasificación Inicial: "Bueno" (Aprobado) y "Malo" (Rechazado)
    is_bueno = is_principal_ok_30 & is_secondary_ok
    df['clasificacion_bueno_malo'] = np.where(is_bueno, 'Bueno', 'Malo')

    # 7. Sub-Clasificación de los "Buenos" en BMB y MMM
    is_bmb = is_principal_ok_36 & is_secondary_ok
    is_mmm = is_bueno & (~is_bmb) # Los aprobados (Buenos) que NO son BMB

    # 8. Clasificación Final
    df['clasificacion_final'] = np.select(
        [is_bmb, is_mmm, is_bueno], # is_bueno se incluye por si acaso, pero is_bmb e is_mmm cubren todos los "Buenos"
        ['BMB', 'MMM', 'Malo'], # Si no es BMB ni MMM, se queda como Malo
        default='Malo'
    )

    # 9. Resultados y Verificaciones
    buenos_count = df['clasificacion_bueno_malo'].value_counts().get('Bueno', 0)
    buenos_pct = (buenos_count / total_clients) * 100

    bmb_count = df['clasificacion_final'].value_counts().get('BMB', 0)
    mmm_count = df['clasificacion_final'].value_counts().get('MMM', 0)
    malo_count = df['clasificacion_final'].value_counts().get('Malo', 0)

    print("\n--- Clasificación Inicial (Buenos/Malos) ---")
    print(f"Total de clientes 'Buenos' (Aprobados): {buenos_count}")
    print(f"Porcentaje de clientes 'Buenos': {buenos_pct:.2f}%")
    print(df['clasificacion_bueno_malo'].value_counts().to_markdown(numalign="left", stralign="left"))

    print("\n--- Clasificación Final (BMB/MMM/Malo) ---")
    print(f"Total BMB + MMM: {bmb_count + mmm_count} (Debe ser igual a 'Buenos': {buenos_count})")
    print(f"Total BMB: {bmb_count}")
    print(f"Total MMM: {mmm_count}")
    print(f"Total Malo: {malo_count}")
    print(df['clasificacion_final'].value_counts().to_markdown(numalign="left", stralign="left"))

    # Guardar el DataFrame con la clasificación final
    df.to_excel(output_file_path, index=False)
    print(f"\nDataFrame con clasificación final guardado en {output_file_path}")

except FileNotFoundError:
    print(f"Error: El archivo {file_path} no fue encontrado.")
except Exception as e:
    print(f"Ocurrió un error al cargar o procesar el archivo: {e}")
