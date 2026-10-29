# Modelo de aforo v2: ocupación a bordo estimada con matriz O-D

**Estado:** piloto en producción para **Metro de Lima Línea 1**. Troncal, corredores y alimentadores
siguen con el índice v1 (presión de abordaje); la API lo indica en `metodo_aforo`.

## 1. Problema

La ATU solo entrega **subidas** (validaciones de tarjeta por estación y hora). El aforo de un tren no
depende solo de cuántos suben en una estación: depende de cuántos **ya vienen a bordo** y de cuántos
**bajan**. El índice v1 (`subidas / capacidad_hora + 0,3 · subidas de 2 estaciones previas`) ignoraba
las bajadas y se clasificaba con percentiles (p60/p90 por franja), así que "Saturado" era relativo y
el "% de aforo" de la app no era ocupación real.

## 2. Método

```
subidas pronosticadas (LightGBM, por estación y hora)
   └─► bajadas estimadas: viaje de retorno
         └─► matriz O-D horaria: gravedad doblemente acotada (Furness)
               └─► carga por tramo y sentido
                     └─► pasajeros por tren (horario oficial)
                           └─► clase por densidad de pie + espera esperada
```

| Paso | Detalle | Referencia |
|---|---|---|
| Bajadas `D_j(h)` | Quien baja en *j* por la mañana vuelve a subir en *j* por la tarde: `D_j(AM) ∝ O_j(PM)`, `D_j(PM) ∝ O_j(AM)`, mediodía con su propio bloque y noche con el día completo. Cada hora cumple `Σ D = Σ O` | Encadenamiento de viajes agregado (propuesta del equipo, validada abajo) |
| Matriz O-D | `T_ij = a_i b_j f(c_ij)` con ceros estructurales (`T_ii = 0`), balanceada por Furness | Ortúzar y Willumsen (2008), §5.2.3 ec. 5.6–5.7, §5.3.2 ec. 5.14–5.18 |
| Fricción | `f(c) = exp(−β c)`, con c = tiempo en vehículo. **β = 0** (ME2: sin distribución de longitudes observada, la matriz más probable es la de máxima entropía) | §12.4.4 ec. 12.21–12.23; calibración de Hyman (§5.5.2) implementada para cuando haya datos |
| Carga | `L_k = Σ_{i≤k<j} T_ij` = subidas acumuladas − bajadas acumuladas, por sentido | Perfil de carga de una línea (§12.5.2, corredores) |
| Pasajeros por tren | `L_k / trenes_por_hora(h)`, con el horario oficial de sept. 2026 (LAB/SAB/DOM, feriados = DOM) | Línea 1 / La República; Infobae |
| Clase | **Asientos** ≤ 187; **De pie** hasta 4 pasajeros/m² de pie (846 por tren); **Saturado** por encima. La capacidad nominal (1200) equivale a 6,15 pasajeros/m², el estándar europeo de aplastamiento | TCQSM (TRB), Tirachini et al. (2013) |
| Espera | `h/2` (llegada aleatoria, servicio regular) + `h · min(1, exceso/subidas por tren)` si la demanda supera la capacidad | §10.6.2.4 ec. 10.19 |

Archivos: `model/od_matrix.py` (algoritmos genéricos), `model/ocupacion.py` (capa por línea),
`models/metro_l1/linea_config.json` (geometría, horario, vehículo, parámetros y fuentes; lo genera
`scripts/construir_linea_metro_l1.py`).

### Datos de la línea

| Dato | Valor | Fuente / estado |
|---|---|---|
| Estaciones y orden | 26, de Villa El Salvador a Bayóvar | GeoJSON del shapefile ATU |
| Posición (km) | Distancias entre estaciones reescaladas a la longitud del trazado: **34,7 km**; Grau en el km 22,0 | Ositrán: 34 km y tramo 1 de 22,1 km |
| Velocidad comercial | 38,6 km/h (34,7 km en 54 min) | Wikipedia / Línea 1 |
| Capacidad por tren | 1200 | AATE / Ositrán |
| Asientos por tren | 187 | **Supuesto:** 156 de la ficha genérica Metropolis 9000 de 5 coches × 6/5 |
| Área de pie | 165 m² (102 m × 2,71 m × 0,90 − 187 × 0,45 m²) | Supuesto de diseño |

## 3. Día de referencia (fechas posteriores a 2025)

Los datos ATU cubren 2025. Para pronosticar 2026 o después, el modelo usa el **día homólogo de 2025**
(`services/calendario.py`), con estas reglas:

- **Día normal:** mismo día de la semana, 52 semanas antes. Si cae en un feriado, una víspera o un
  paro de 2025, se desplaza ±1 semana.
- **Feriado:** el mismo feriado de 2025. Semana Santa se calcula desde la fecha de Pascua. Si ese
  feriado no tiene 28 días de historia previa, se usa el feriado de 2025 más parecido.
- **Víspera** (24 y 31 de diciembre): la misma víspera de 2025.

La respuesta trae `referencia.fecha_referencia_datos` y la fecha y hora devueltas son las del día
pedido, en hora de Lima. Ya no se fija la fecha de hoy con el minuto actual.

## 4. Modelo de demanda en producción: LightGBM

| Sistema | WMAPE test LightGBM | WMAPE test ensemble | WMAPE validación (ens / LGB) |
|---|---:|---:|---:|
| Metro L1 | **0,0996** | 0,1021 | 0,0723 / 0,0755 |
| Troncal | **0,1150** | 0,1229 | 0,1032 / 0,1059 |
| Corredores | **0,1132** | 0,1196 | 0,0948 / 0,0968 |
| Alimentadores | 0,4017 | 0,4017 (es LightGBM) | — |

El ensemble con Chronos gana en validación por un 3–4 % relativo, pero en el test retenido LightGBM
iguala o supera al ensemble en todos los sistemas. Además, el ensemble tarda 52 s por día en Metro L1
frente a milisegundos de LightGBM, y necesita torch, que no entra en la instancia de Render. Por eso
**LightGBM es el modelo de producción**: es una decisión explícita, ya no un modo "ligero" de
respaldo.

## 5. Evaluación (Metro L1)

Reproducible con `python scripts/evaluar_od_metro_l1.py`; los resultados quedan en
`models/metro_l1/od_evaluacion.json`.

### 5.1 Recuperación sintética

Son 40 simulaciones de días laborables reales. La matriz "verdadera" se genera con atracciones con
ruido lognormal (σ = 0,35) y un β verdadero ~ U(0, 0,06), es decir, violando a propósito los supuestos
del estimador. El estimador solo ve las subidas.

| Métrica (media) | O-D con viaje de retorno | Baseline: bajadas uniformes |
|---|---:|---:|
| WAPE de la carga por tramo y hora | **0,237** | 0,327 |
| WAPE de las bajadas | **0,261** | 0,632 |
| Accuracy de clase | **0,871** | 0,812 |
| F1-macro de clase | **0,715** | 0,552 |
| Error relativo de la carga pico | 0,155 | 0,098 |

Con β = 0 la carga pico queda sesgada hacia arriba (viajes más largos que con β > 0). El sesgo es
conservador, porque avisa saturación de más y no de menos. La sensibilidad está en la tabla 5.2.

### 5.2 Sensibilidad a β (promedio de días laborables, mar–oct 2025)

| β (1/min) | Viaje medio | Asimetría de la matriz diaria | Pico AM / PM (pasajeros por tren) |
|---:|---:|---:|---:|
| 0,00 | 20,2 min / 13,0 km | **0,065** | 904 / 1007 |
| 0,02 | 18,2 min / 11,7 km | 0,066 | 839 / 936 |
| 0,05 | 15,4 min / 9,9 km | 0,067 | 759 / 842 |
| 0,08 | 13,3 min / 8,5 km | 0,070 | 707 / 779 |

La matriz diaria estimada es casi simétrica (`T_ij ≈ T_ji` en un 93,5 %), coherente con la
hipótesis de viaje de retorno. β = 0 es la que minimiza la asimetría.

### 5.3 Contraste con datos públicos (día laborable típico)

| Hora | Sentido | Tramo más cargado | Pasajeros por tren | Ocupación |
|---|---|---|---:|---:|
| 07:00 | hacia Bayóvar | Ayacucho → Cabitos | 904 | 75 % |
| 07:00 | hacia Villa El Salvador | El Ángel → Miguel Grau | 887 | 74 % |
| 18:00 | hacia Bayóvar | Miguel Grau → El Ángel | 1007 | 84 % |
| 18:00 | hacia Villa El Salvador | Ayacucho → Jorge Chávez | 927 | 77 % |

- **Patrón espacial:** en la mañana los viajes van de San Juan de Lurigancho y Villa El Salvador
  hacia el centro (Grau, Gamarra, La Cultura, Angamos) y en la tarde vuelven. Es el patrón conocido
  de la Línea 1.
- **Nivel de carga:** son promedios horarios con un tren cada 3 minutos. La punta de 15 minutos es
  mayor, lo que concuerda con el operador ("en hora punta los trenes van a su máxima capacidad").
- **Demanda:** 600 mil subidas por día laborable y 201,5 millones en 2025. Ositrán reporta 191,1
  millones en 2024.

### 5.4 Pronóstico en test (2025-11-01 a 2025-12-31, 61 días)

La "verdad" es la ocupación calculada con las subidas observadas por la misma cadena O-D. Se mide
cuánto del error de ocupación aporta el pronóstico.

| Modelo | MAE (pasajeros por tren) | WMAPE de la carga | Accuracy | F1-macro | F1-weighted |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **22,3** | **0,049** | **0,960** | **0,922** | **0,960** |
| media_4sem (baseline) | 58,2 | 0,127 | 0,918 | 0,830 | 0,917 |

- **RNF-01** (accuracy ≥ 0,75 y F1-macro ≥ 0,72): **cumple**.
- **Distribución de clases observadas:** De pie 75 %, Asientos 17 %, Saturado 8 %.
- **Latencia:** pronóstico más O-D en 0,02 s por día. En la API, P95 en caliente < 2 s (AC-04, ver
  `tests/test_api.py`).
- **Regla U:** el acuerdo es 97,7 %, pero "siempre abordar" ya acierta 96,7 %, así que la métrica
  aporta poco y hay que leerla con cuidado.

## 6. Limitaciones

- **Sin datos de bajadas ni GPS:** la matriz O-D no se puede validar contra viajes reales. La
  validación es sintética, por consistencia interna y por contraste con cifras públicas. El
  aforo percibido de `feedback_usuario` y el botón de reporte permitirán recalibrar.
- **Promedios horarios:** no se modela la punta de 15 minutos (factor de hora punta) ni la
  irregularidad del intervalo.
- **Viajes que cruzan de hora:** un viaje se asigna a la hora en que sube.
- **Asientos supuestos:** el número de asientos por tren de Lima es una estimación.
- **β sin calibrar:** no hay una distribución de longitudes de viaje observada; con una encuesta
  O-D o con datos de tarjeta con destino, se calibraría con Hyman (`od_matrix.calibrar_hyman`).
- **Solo Metro L1:** el Metropolitano tiene servicios expresos y un ramal en el centro (no es una
  línea simple), y corredores y alimentadores necesitan pronóstico por paradero. Quedan como
  siguiente paso.

## 7. Fuentes

- Ortúzar, J. de D. y Willumsen, L. G. (2008). *Modelos de Transporte*. PUbliCan, Universidad de Cantabria.
- Ositrán, Informe de Desempeño 2024, Línea 1: <https://www.ositran.gob.pe/anterior/wp-content/uploads/2018/04/id-2024-linea1.pdf>
- Ositrán, Informe de Desempeño 2022, Línea 1: <https://www.ositran.gob.pe/anterior/wp-content/uploads/2023/07/id-l1-2022.pdf>
- Horario oficial sept. 2026: <https://larepublica.pe/sociedad/2026/09/04/linea-1-del-metro-de-lima-oficializa-horarios-de-atencion-para-septiembre-de-2026-207920>
- Domingos y feriados: <https://www.infobae.com/peru/2026/10/18/horarios-del-tren-de-lima-frecuencia-de-paso-en-hora-punta-fines-de-semana-y-claves-para-organizar-el-viaje/>
- Capacidad de 1200 por tren: <https://gestion.pe/economia/metro-lima-incorpora-primer-tren-6-coches-capacidad-1-200-pasajeros-245920-noticia/>
- Alstom Metropolis 9000: <https://es.wikipedia.org/wiki/Alstom_Metropolis_9000>
- Línea 1 (longitud, tiempo de extremo a extremo): <https://es.wikipedia.org/wiki/L%C3%ADnea_1_del_Metro_de_Lima_y_Callao>
- TCQSM 2.ª ed., Parte 5 (densidades de pie): <https://onlinepubs.trb.org/onlinepubs/tcrp/docs/tcrp100/Part5.pdf>
- Li, Z. y Hensher, D. (2013). Crowding in Public Transport: A Review of Objective and Subjective Measures: <https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=1065&context=jpt>
