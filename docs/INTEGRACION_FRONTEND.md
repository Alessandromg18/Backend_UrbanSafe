# Cambios necesarios en el frontend (modelo de aforo v2)

La API actual sigue siendo compatible: las claves que lee la app (`crowding.index`, `crowding.class`,
`eta.next`, `eta.second`, `recomendacion`, `predictions[]`) no cambiaron. Aun así, hay tres problemas
que solo se corrigen en el frontend.

## 1. Quitar el "Naranjal proxy": usar `model_series` del backend

`mapStationToModelSeries` (`src/api/services.ts`) devuelve por defecto `Naranjal`, `301`, `Angamos` o
`Estacion Central` cuando no encuentra coincidencia. Así la app muestra la predicción de otra estación
como si fuera la del paradero elegido.

Ahora `GET /api/stations` trae en cada estación `model_system` y `model_series`. `model_series` es
`null` cuando el paradero no tiene datos ATU propios. Además existe un endpoint que resuelve todo por
id y devuelve el estado actual y el día completo **en una sola llamada** (antes eran dos):

```ts
// src/api/services.ts
export const getStationPredictionsById = async (stationId: string, hour: number, dia: string, haciaId?: string) => {
  const res = await api.get(`/predictions/estacion/${encodeURIComponent(stationId)}`, {
    params: { dia, hour, hacia_id: haciaId },
  });
  return { current: res.data.current, forecast: res.data.forecast, meta: res.data };
};
```

Si responde **404** con `"Sin predicción"`, la app debe mostrar "Sin datos de demanda para este
paradero" en lugar de inventar una predicción.

## 2. Fecha en hora de Lima, no en UTC

`new Date().toISOString().split('T')[0]` devuelve la fecha **UTC**: desde las 19:00 de Lima ya es
"mañana", y la app pide el pronóstico del día siguiente. Lo más simple es **no enviar `dia`**: el
backend usa la fecha y la hora de Lima por defecto. Si se necesita enviarla:

```ts
const hoyLima = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Lima' }).format(new Date()); // YYYY-MM-DD
```

## 3. Sentido de viaje: aforo en la dirección correcta

En Metro L1 el aforo depende del sentido: hacia Bayóvar o hacia Villa El Salvador. Cuando el usuario
ya eligió una ruta (`match-trip` devuelve `boarding_id` y `leaving_id`), hay que pasar el destino:

```ts
getStationPredictionsById(route.boarding_id, hora, hoyLima, route.leaving_id)
```

Si no se pasa destino, el backend informa el **sentido más cargado** (criterio conservador) en
`current.sentido`, y el detalle de ambos sentidos en `current.por_sentido`.

## Campos nuevos que conviene mostrar

| Campo | Significado |
|---|---|
| `crowding` | Ocupación real = pasajeros a bordo / capacidad nominal del tren (puede superar 1: la demanda supera la oferta) |
| `pasajeros_por_tren`, `sentido` | Para el texto "≈ 900 pasajeros por tren hacia Bayóvar" |
| `class` = `Sin_servicio`, `recommendation` = `sin_servicio` | Horas sin trenes (p. ej. 23:00–04:59): mostrar "Fuera de horario" |
| `eta_min` | Espera esperada en el andén (intervalo/2 con el horario oficial), no la llegada de un tren rastreado |
| `referencia.fecha_referencia_datos` | Día de 2025 con datos ATU usado como base; conviene mostrarlo en letra pequeña |
| `metodo_aforo` | `ocupacion_od_v2` (Metro L1) o `presion_abordaje_v1` (otros sistemas, aún no es ocupación real) |
