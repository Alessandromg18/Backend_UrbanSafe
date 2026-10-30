# UrbanSafe Backend API

API REST y motor de inferencia analítica desarrollado para **UrbanSafe**, responsable de gestionar la autenticación de usuarios (JWT), procesar el proxy geoespacial de OpenStreetMap y ejecutar los modelos predictivos de Machine Learning (LightGBM) para la afluencia en el transporte público de Lima Metropolitana.

## Modelo de predicción (v2)

- **Demanda:** LightGBM por sistema, pronóstico horario de subidas (validaciones ATU); en alimentadores, por ruta × sentido (v1.2.0). Es el modelo de producción: en el test retenido iguala o supera al ensemble con Chronos y corre en milisegundos. Las métricas de test son fuera de muestra (reentrenado solo con días previos).
- **Ocupación a bordo (los 4 sistemas):** matriz origen-destino estimada solo con subidas, con bajadas por viaje de retorno. Metro L1 y troncal: línea doblemente acotada (Furness). Corredores: por ruta y sentido a nivel de paradero (acotada a orígenes). Alimentadores: carga crítica junto a la estación de integración. Los pasajeros por vehículo salen del horario oficial (Metro) o de la flota publicada (buses). Clases por densidad de pie: Asientos, De pie, Saturado. Detalle, evaluación en test y fuentes en [docs/MODELO_OCUPACION.md](docs/MODELO_OCUPACION.md).
- **Fechas:** hora de Lima. Para fechas posteriores a 2025 se usa el día homólogo de 2025 con datos ATU (mismo día de la semana y tipo de día), declarado en `referencia`.
- **Endpoints:**

| Endpoint | Uso |
|---|---|
| `GET /api/predictions/estacion/<station_id>?dia&hour&hacia_id` | Estado actual + 24 h de un paradero en una sola llamada (recomendado) |
| `GET /api/predictions/<sistema>/<serie>?dia&hour&sentido\|hacia` | Una hora |
| `GET /api/predictions/<sistema>/<serie>/forecast?dia&hours&desde_hora&sentido\|hacia` | Horas del día |
| `GET /api/stations` | Cada estación trae `model_system` y `model_series` (`null` = sin datos ATU, no se inventa la predicción) |

- **Integración con el frontend:** [docs/INTEGRACION_FRONTEND.md](docs/INTEGRACION_FRONTEND.md) (implementada en la rama `feat/aforo-v2-integracion` del frontend).

## Arquitectura

Esta API (Flask, desplegada en Render) es **la arquitectura del producto**. No hay base de datos de
demanda: los modelos y su contexto viajan como archivos versionados.

```
App (Expo) ──HTTP──► Flask API
                      ├─ routes/        auth (SQLite, solo usuarios), stations, geo, predictions
                      ├─ services/      model_service (caché de modelos y pronósticos), calendario
                      │                 (hora de Lima, feriados, día homólogo), mapeo_estaciones
                      ├─ model/         inferencia (LightGBM), od_matrix (Furness/gravedad),
                      │                 ocupacion (capas por sistema)
                      ├─ models/        artefactos versionados + manifest.json (sha256)
                      └─ data/          context/<sistema>_2025.parquet (validaciones ATU), geo/
```

El repositorio del curso (`EV081/DPD_project`, carpeta `deliveries/week10`) contiene además un
flujo con PostgreSQL (`rutas_directas()`, tabla `prediccion`, ajuste con TomTom, `feedback_usuario`)
y el pipeline de ingesta y entrenamiento. **La app no usa ese flujo de base de datos**: lo reemplaza
esta API con archivos versionados. El pipeline de entrenamiento y los datos limpios del repo del
curso sí son la fuente de los artefactos de `models/` y de `data/context/`.
- **Tests:** `pip install pytest && python -m pytest tests` (algoritmo de Furness contra la Tabla 5.7 de Ortúzar y Willumsen, calendario, API y latencia P95).
- **Versión de modelos: 1.3.0** (`models/manifest.json`; la API verifica los sha256 al cargar).
- **Regenerar o evaluar:** `python scripts/construir_linea_metro_l1.py`, `python scripts/construir_linea_troncal.py`, `python scripts/construir_rutas_bus.py --datos <DPD_project>/deliveries/week10/data_processed/clean`, `python scripts/entrenar_alimentador_sentido.py --datos <...>/clean`, `python scripts/evaluar_od_metro_l1.py`, `python scripts/evaluar_ocupacion.py` y `python scripts/serializar_modelos.py <versión> "<cambios>"`.


## Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Framework Web:** Flask
* **Machine Learning:** LightGBM, Scikit-Learn, NumPy (matriz O-D)
* **Control de Tráfico / Geo:** Requests, OpenStreetMap Nominatim API Integration


## Configuración y Ejecución Local

### 1. Requisitos Previos
* Tener instalado **Python 3.10 o superior** y `pip`.

### 2. Clonar el Repositorio e Instalar Dependencias
Abre tu terminal en la carpeta del backend y ejecuta los siguientes comandos:

```bash
# Instalar las librerías y dependencias necesarias
pip install -r requirements.txt
```
### 3. Configurar Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto (o configúralo en tu entorno local) con la siguiente llave de seguridad para los tokens de acceso.

```bash
JWT_SECRET_KEY=super-secret-key-urban-safe-2026-xyz-987
```
### 4. Ejecutar la aplicacióm
Inicia el servidor local ejecutando el archivo principal:

```bash
python app.py
```
El servidor se levantará por defecto en `http://localhost:5000` o el puerto configurado).

---

## Instrucciones de Despliegue (Deployment en Render)

El backend está configurado para desplegarse de manera automatizada en la plataforma en la nube **Render**:

### 1. Conectar el Repositorio

Conecta el repositorio de GitHub (https://github.com/Alessandromg18/Backend_UrbanSafe.git) a un nuevo **Web Service** en Render.

### 2. Configurar el Comando de Inicio (`Procfile`)

El repositorio incluye un archivo `Procfile` en la raíz que indica a Render cómo arrancar la aplicación de Python de forma automática.

### 3. Variables de Entorno en Producción

En el panel de configuración de Render (**Environment**), añade la siguiente variable obligatoria:

```env
JWT_SECRET_KEY=super-secret-key-urban-safe-2026-xyz-987
```

### 4. Despliegue Automático

Cada vez que realices un `git push` a la rama principal, Render recompilará las dependencias a partir de `requirements.txt` y actualizará el servicio en línea.
