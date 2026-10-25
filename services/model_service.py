from pathlib import Path
import pandas as pd
from model.inferencia import ModeloSistema

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "context"

CONTEXT_FILES = {
    "metro_l1": "metro_l1_context.csv",
    "troncal": "troncal_context.csv",
    "corredor": "corredor_context.csv",
    "alimentador": "alimentador_context.csv",
}

class ModelService:
    @staticmethod
    def cargar_contexto(sistema: str) -> pd.DataFrame:
        """Carga el CSV de contexto correspondiente al sistema de transporte."""
        if sistema not in CONTEXT_FILES:
            raise ValueError(f"Sistema no válido: {sistema}. Opciones: {list(CONTEXT_FILES.keys())}")
        
        csv_path = DATA_DIR / CONTEXT_FILES[sistema]
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de contexto en {csv_path}")
        
        df = pd.read_csv(csv_path)
        if "Fecha_Hora" in df.columns:
            df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"])
            df.set_index("Fecha_Hora", inplace=True)
            
        return df

    @classmethod
    def obtener_prediccion(cls, sistema: str, dia: str | None, modo: str = "ligero", unidad: str | None = None) -> list[dict]:
        """Carga el modelo del sistema, procesa el contexto y genera el pronóstico de 24 horas."""
        
        # 1. Cargamos el contexto primero para conocer los límites reales de las fechas disponibles
        contexto = cls.cargar_contexto(sistema)
        min_date = contexto.index.min() + pd.Timedelta(days=28) # Exige los 28 días de historia del modelo
        max_date = contexto.index.max()
        
        # 2. Validación de seguridad anti-nulos o valores vacíos
        if not dia or str(dia).strip() in ["", "undefined", "null", "None"]:
            dia = max_date.strftime("%Y-%m-%d")
            
        try:
            parsed_dia = pd.to_datetime(dia)
            if pd.isna(parsed_dia):
                parsed_dia = max_date
        except Exception:
            parsed_dia = max_date
        
        # 3. Mapeo inteligente por Día de la Semana (DOW) dentro del rango válido del contexto
        if parsed_dia.year > 2025 or parsed_dia > max_date or parsed_dia < min_date:
            target_dow = parsed_dia.dayofweek  # 0=Lunes, ..., 5=Sábado, 6=Domingo
            
            rango_fechas = pd.date_range(min_date, max_date)
            candidatos = [d for d in rango_fechas if d.dayofweek == target_dow]
            
            if candidatos:
                # Seleccionamos el candidato más cercano al día del mes solicitado
                dia_mapeado = min(candidatos, key=lambda d: abs(d.day - parsed_dia.day))
                dia = dia_mapeado.strftime("%Y-%m-%d")
            else:
                dia = max_date.strftime("%Y-%m-%d")
        else:
            dia = parsed_dia.strftime("%Y-%m-%d")

        try:
            modelo = ModeloSistema.cargar(sistema, verificar_hash=False)
        except Exception as e:
            raise RuntimeError(f"Error al cargar los artefactos del modelo para {sistema}: {str(e)}")
        
        df_preds = modelo.pronosticar(contexto, dia, modo=modo)
        
        if unidad:
            df_preds = df_preds[df_preds["unidad"] == unidad]
            if df_preds.empty:
                raise ValueError(f"La unidad o estación '{unidad}' no existe en el sistema '{sistema}'")
                
        return df_preds.to_dict(orient="records")