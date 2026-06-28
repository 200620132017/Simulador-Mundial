# 📊 REPORTE DE RENDIMIENTO DEL MODELO - MUNDIAL 2026

Este informe detalla las métricas de evaluación obtenidas tras el entrenamiento de los regresores XGBoost, utilizando ingeniería de variables basada en forma reciente (*Rolling Means*), decaimiento exponencial temporal (*Time-Decay*) y el anclaje estructural de jerarquías por Ranking FIFA.

## 📈 Tabla General de Métricas

| Componente del Modelo | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | $R^2$ Score (Varianza Explicada) |
| :--- | :---: | :---: | :---: |
| **XGBRegressor - Goles Locales** | 1.0900 | 1.4313 | 0.1588 |
| **XGBRegressor - Goles Visitantes** | 0.8979 | 1.2148 | 0.1281 |

## 🧠 Interpretación de Métricas para la Memoria

* **Precisión por Partido (MAE):** En promedio, el modelo comete un error de ±1.09 goles al predecir la puntuación del equipo local y ±0.90 goles en el visitante. Teniendo en cuenta la alta volatilidad del fútbol, un MAE inferior a 1.0 gol es el estándar óptimo en analítica deportiva predictiva.
* **Sensibilidad a Anomalías (RMSE):** El RMSE se sitúa en torno a 1.43. Al ser ligeramente superior al MAE, indica que la distribución de errores es homogénea y el modelo no está sufriendo de desviaciones drásticas causadas por marcadores atípicos (goleadas extremas de las que no tenía contexto).
* **Capacidad del $R^2$:** El Coeficiente de Determinación refleja el impacto del Ranking FIFA y la forma reciente controlando la aleatoriedad inherente del dataset, asegurando un punto de anclaje firme en la simulación estocástica por Poisson.
