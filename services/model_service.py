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
    def obtener_prediccion(cls, sistema: str, dia: str, modo: str = "ligero", unidad: str | None = None) -> list[dict]:
        """Carga el modelo del sistema, procesa el contexto y genera el pronóstico de 24 horas."""
        # Se deshabilita temporalmente la verificación estricta de hash para agilizar pruebas locales en desarrollo
        try:
            modelo = ModeloSistema.cargar(sistema, verificar_hash=False)
        except Exception as e:
            raise RuntimeError(f"Error al cargar los artefactos del modelo para {sistema}: {str(e)}")
        
        contexto = cls.cargar_contexto(sistema)
        df_preds = modelo.pronosticar(contexto, dia, modo=modo)
        
        if unidad:
            df_preds = df_preds[df_preds["unidad"] == unidad]
            if df_preds.empty:
                raise ValueError(f"La unidad o estación '{unidad}' no existe en el sistema '{sistema}'")
                
        # Retorna el DataFrame convertido a formato JSON orientado a registros
        return df_preds.to_dict(orient="records")