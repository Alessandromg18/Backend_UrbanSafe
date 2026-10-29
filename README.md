# UrbanSafe Backend API

API REST y motor de inferencia analítica desarrollado para **UrbanSafe**, responsable de gestionar la autenticación de usuarios (JWT), procesar el proxy geoespacial de OpenStreetMap y ejecutar los modelos predictivos de Machine Learning (LightGBM) para la afluencia en el transporte público de Lima Metropolitana.

## Modelo de predicción (v2)

- **Demanda:** LightGBM por sistema, pronóstico horario de subidas (validaciones ATU). Es el modelo de producción: en el test retenido iguala o supera al ensemble con Chronos y corre en milisegundos.
- **Ocupación a bordo (Metro L1):** matriz origen-destino estimada solo con subidas (gravedad doblemente acotada / Furness, bajadas por viaje de retorno), carga por tramo y sentido, y pasajeros por tren según el horario oficial. Clases por densidad de pie: Asientos, De pie, Saturado. Detalle, evaluación y fuentes en [docs/MODELO_OCUPACION.md](docs/MODELO_OCUPACION.md).
- **Fechas:** hora de Lima. Para fechas posteriores a 2025 se usa el día homólogo de 2025 con datos ATU (mismo día de la semana y tipo de día), declarado en `referencia`.
- **Endpoints:**

| Endpoint | Uso |
|---|---|
| `GET /api/predictions/estacion/<station_id>?dia&hour&hacia_id` | Estado actual + 24 h de un paradero en una sola llamada (recomendado) |
| `GET /api/predictions/<sistema>/<serie>?dia&hour&sentido\|hacia` | Una hora |
| `GET /api/predictions/<sistema>/<serie>/forecast?dia&hours&desde_hora&sentido\|hacia` | Horas del día |
| `GET /api/stations` | Cada estación trae `model_system` y `model_series` (`null` = sin datos ATU, no se inventa la predicción) |

- **Cambios pendientes en el frontend:** [docs/INTEGRACION_FRONTEND.md](docs/INTEGRACION_FRONTEND.md).
- **Tests:** `pip install pytest && python -m pytest tests` (algoritmo de Furness contra la Tabla 5.7 de Ortúzar y Willumsen, calendario, API y latencia P95).
- **Regenerar o evaluar:** `python scripts/construir_linea_metro_l1.py` y `python scripts/evaluar_od_metro_l1.py`.


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
