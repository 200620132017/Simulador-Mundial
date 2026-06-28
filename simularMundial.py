import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson

# 1. Cargar modelos entrenados de 6 dimensiones (goles + ranking)
try:
    model_home = joblib.load('model_home.pkl')
    model_away = joblib.load('model_away.pkl')
except FileNotFoundError:
    print("Error: Asegúrate de tener 'model_home.pkl' y 'model_away.pkl' en el directorio.")
    exit()

# 2. Cargar base de datos histórica reciente
df = pd.read_csv('results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= 2014].sort_values('date', ascending=False)

# Fijamos semilla para mantener reproducibilidad en las tandas de penaltis
np.random.seed(42)

# --- DICCIONARIO MAESTRO: RANKING FIFA OFICIAL (MUNDIAL 2026) ---
RANKING_FIFA_2026 = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4, "Brazil": 5,
    "Belgium": 6, "Portugal": 7, "Netherlands": 8, "Uruguay": 9, "Italy": 10,
    "Colombia": 12, "Germany": 13, "Morocco": 14, "Mexico": 15, "United States": 16,
    "Japan": 18, "Switzerland": 19, "Iran": 20, "Senegal": 21, "Sweden": 22,
    "South Korea": 23, "Austria": 25, "Ukraine": 27, "Australia": 28, "Turkey": 30,
    "Ecuador": 31, "Egypt": 32, "Croatia": 35, "Norway": 40, "Algeria": 44,
    "Czech Republic": 45, "Tunisia": 46, "Ivory Coast": 48, "Saudi Arabia": 53,
    "Ghana": 60, "Iraq": 62, "South Africa": 65, "Cape Verde": 68, "Panama": 71,
    "Bosnia and Herzegovina": 74, "Uzbekistan": 76, "Qatar": 80, "Jordan": 82, 
    "Haiti": 86, "Curacao": 90, "Oman": 92, "DR Congo": 95, "New Zealand": 105, "Slovakia": 32
}

def obtener_stats_reales(team, df, n_partidos=8):
    partidos_equipo = df[(df['home_team'] == team) | (df['away_team'] == team)].head(n_partidos)
    if partidos_equipo.empty:
        return None
    goles_favor, goles_contra = [], []
    for _, row in partidos_equipo.iterrows():
        if row['home_team'] == team:
            goles_favor.append(row['home_score'])
            goles_contra.append(row['away_score'])
        else:
            goles_favor.append(row['away_score'])
            goles_contra.append(row['home_score'])
    return {
        'favor': np.nanmean(goles_favor) if goles_favor else 1.0,
        'contra': np.nanmean(goles_contra) if goles_contra else 1.0
    }

def simular_partido_eliminatoria(team1, team2):
    stats_t1 = obtener_stats_reales(team1, df)
    stats_t2 = obtener_stats_reales(team2, df)
    
    if not stats_t1 or not stats_t2:
        return None
        
    rank_t1 = RANKING_FIFA_2026.get(team1, 70.0)
    rank_t2 = RANKING_FIFA_2026.get(team2, 70.0)
    
    features = np.array([[
        stats_t1['favor'], stats_t1['contra'], 
        stats_t2['favor'], stats_t2['contra'],
        rank_t1, rank_t2
    ]])
    
    # 1. Predicción base de goles esperados según el modelo entrenado
    lambda_home_base = model_home.predict(features)[0]
    lambda_away_base = model_away.predict(features)[0]
    
    # 2. INTRODUCIMOS EL AZAR CONTROLADO: Un pequeño factor de ruido (desviación de 0.25 goles)
    # Esto simula un mal día, un tiro al palo o una mala decisión arbitral sin romper la jerarquía.
    ruido_home = np.random.normal(0, 0.25)
    ruido_away = np.random.normal(0, 0.25)
    
    lambda_home = max(0.1, lambda_home_base + ruido_home)
    lambda_away = max(0.1, lambda_away_base + ruido_away)
    
    # 3. Calculamos la matriz de Poisson con las tasas ligeramente alteradas por el azar
    max_goles = 6
    prob_home = [poisson.pmf(i, lambda_home) for i in range(max_goles)]
    prob_away = [poisson.pmf(j, lambda_away) for j in range(max_goles)]
    matriz_resultados = np.outer(prob_home, prob_away)
    
    # Seleccionamos el marcador más probable bajo este escenario con ruido
    goles_t1, goles_t2 = np.unravel_index(np.argmax(matriz_resultados), matriz_resultados.shape)
    
    # Lógica de resolución en caso de empate reglamentario
    if goles_t1 > goles_t2:
        ganador = team1
        nota = ""
    elif goles_t2 > goles_t1:
        ganador = team2
        nota = ""
    else:
        # En caso de empate, los penaltis siguen siendo un 50/50 estocástico
        ganador = np.random.choice([team1, team2], p=[0.5, 0.5])
        nota = f" (Gana {ganador} en penaltis)"
        
    return {
        'marcador': f"{goles_t1} - {goles_t2}",
        'ganador': ganador,
        'texto': f"⚽ {team1} vs {team2} -> Marcador: {goles_t1} - {goles_t2}{nota}"
    }

def emparejar_ganadores(lista_ganadores):
    # Toma una lista de ganadores y los agrupa de dos en dos de forma correlativa para la siguiente fase
    return [(lista_ganadores[i], lista_ganadores[i+1]) for i in range(0, len(lista_ganadores), 2)]

def procesar_fase(parejas, nombre_fase, f_out):
    print(f"\n🏆 SIMULANDO: {nombre_fase} 🏆")
    f_out.write(f"\n--- {nombre_fase} ---\n")
    ganadores = []
    perdedores = []
    
    for t1, t2 in parejas:
        res = simular_partido_eliminatoria(t1, t2)
        if not res:
            print(f" ❌ Error al procesar o faltan datos: {t1} vs {t2}")
            # Si faltan datos en el CSV, pasa por defecto el de mejor ranking para no romper la cascada
            ganador_emergencia = t1 if RANKING_FIFA_2026.get(t1, 70) < RANKING_FIFA_2026.get(t2, 70) else t2
            ganadores.append(ganador_emergencia)
            perdedores.append(t2 if ganador_emergencia == t1 else t1)
            continue
            
        print(res['texto'])
        f_out.write(res['texto'] + "\n")
        ganadores.append(res['ganador'])
        perdedores.append(t1 if res['ganador'] == t2 else t2)
        
    return ganadores, perdedores

# =====================================================================
# CONFIGURACIÓN DE LOS DIECISEISEAVOS DE FINAL (Cuadro Vertical Completo)
# =====================================================================
# Bloque A (Mitad Izquierda del Póster)
dieciseisavos_izq = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal")
]

# Bloque B (Mitad Derecha del Póster)
dieciseisavos_der = [
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Argentina", "Cape Verde"), 
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana")
]

# =====================================================================
# EJECUCIÓN DEL TORNEO EN CASCADA MATEMÁTICA PURA
# =====================================================================
with open("simulacion_fase_final.txt", "w", encoding="utf-8") as f_out:
    f_out.write("==================================================\n")
    f_out.write("🔮 SIMULACIÓN DEL MUNDIAL DE 48 SELECCIONES (ANCLAJE DE RANKING) 🔮\n")
    f_out.write("==================================================\n")

    # 1. DIECISEISEAVOS DE FINAL (Ronda de 32)
    ganadores_16_izq, _ = procesar_fase(dieciseisavos_izq, "DIECISEISEAVOS DE FINAL - BLOQUE A", f_out)
    ganadores_16_der, _ = procesar_fase(dieciseisavos_der, "DIECISEISEAVOS DE FINAL - BLOQUE B", f_out)

    # 2. OCTAVOS DE FINAL (Ronda de 16)
    cruces_octavos_izq = emparejar_ganadores(ganadores_16_izq)
    cruces_octavos_der = emparejar_ganadores(ganadores_16_der)
    
    ganadores_8_izq, _ = procesar_fase(cruces_octavos_izq, "OCTAVOS DE FINAL - LLAVE IZQUIERDA", f_out)
    ganadores_8_der, _ = procesar_fase(cruces_octavos_der, "OCTAVOS DE FINAL - LLAVE DERECHA", f_out)

    # 3. CUARTOS DE FINAL (Ronda de 8)
    cruces_cuartos_izq = emparejar_ganadores(ganadores_8_izq)
    cruces_cuartos_der = emparejar_ganadores(ganadores_8_der)
    
    ganadores_4_izq, _ = procesar_fase(cruces_cuartos_izq, "CUARTOS DE FINAL - LLAVE IZQUIERDA", f_out)
    ganadores_4_der, _ = procesar_fase(cruces_cuartos_der, "CUARTOS DE FINAL - LLAVE DERECHA", f_out)

    # 4. SEMIFINALES
    # Se cruzan los supervivientes de la izquierda por un lado y la derecha por el otro
    cruces_semis = [
        (ganadores_4_izq[0], ganadores_4_izq[1]),
        (ganadores_4_der[0], ganadores_4_der[1])
    ]
    ganadores_semis, perdedores_semis = procesar_fase(cruces_semis, "SEMIFINALES", f_out)

    # 5. TERCER Y CUARTO PUESTO
    _, _ = procesar_fase([(perdedores_semis[0], perdedores_semis[1])], "PARTIDO POR EL TERCER PUESTO (BRONZE FINAL)", f_out)

    # 6. GRAN FINAL
    print("\n👑 LA GRAN FINAL DEL MUNDIAL 2026 👑")
    f_out.write("\n👑 LA GRAN FINAL DEL MUNDIAL 2026 👑\n")
    resultado_final = simular_partido_eliminatoria(ganadores_semis[0], ganadores_semis[1])
    
    print(f"🔥 RESULTADO: {resultado_final['texto']}")
    print(f"🏆 ¡CAMPEÓN DEL MUNDO 2026: {resultado_final['ganador'].upper()}! 🏆\n")
    f_out.write(f"FINAL: {resultado_final['texto']}\n")
    f_out.write(f"🏆 CAMPEÓN DEL MUNDO: {resultado_final['ganador'].upper()} 🏆\n")

print("-> Reporte jerárquico real exportado a 'simulacion_fase_final.txt'.")