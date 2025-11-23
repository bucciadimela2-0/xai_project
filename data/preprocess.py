import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

FREQ_REQUIRED = 4_000_000.0
TARGET_COL = "FM_data_peak_distorted_echo_power"
DROP_COLS = [
    "FM_data_ephemeris_time",
    "FM_data_median_corrected_echo_power",
    "FM_data_orbit_number",
    "FM_data_peak_corrected_echo_power",
    "FM_data_peak_simulated_echo_power",
    "FM_data_solar_longitude",
]
FLUX_COL = "FM_data_F10_7_index"
FREQ_COL = "FM_data_frequency"

#seleziona solo le righe con FM_data_frequency == freq 
def filter_data(df: pd.DataFrame, freq: float = FREQ_REQUIRED, cancel_column: bool = True ) -> pd.DataFrame:
    if FREQ_COL not in df.columns:
        raise ValueError(f"Required column absent: {FREQ_COL} ")
    filtered = df[df[FREQ_COL] == freq].copy()
    if filtered.empty:
        raise ValueError(f"No rows with {FREQ_COL} == {freq}")
    if cancel_column:
        filtered.drop(columns=[FREQ_COL],inplace=True )

    return filtered



def clean_data(df: pd.DataFrame, drop_cols: Optional[list] = DROP_COLS, keep_flux = False):
    
    cleaned = df.drop(columns=drop_cols, errors="ignore")
    if not keep_flux and FLUX_COL in cleaned.columns:
        cleaned = cleaned.drop(columns = [FLUX_COL])
    return cleaned
#Divide i file in train, val e test
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: str = "none",
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if shuffle not in ["none", "all", "train_only"]:
        raise ValueError("shuffle deve essere 'none', 'all' o 'train_only'")

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True) if shuffle == "all" else df.copy()

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if shuffle == "train_only":
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df



#Pipeline pre-processo
def preprocess(
    input_path: str,
    output_dir: str = "data/data_cleaned",
    keep_flux: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: str = "none",
    random_state: int = 42,
    
) -> None:
    print(f"[INFO] Caricamento: {input_path}")
    df = pd.read_csv(input_path, sep=";")
    print(f"[INFO] Dataset originale: {df.shape[0]} righe, {df.shape[1]} colonne")

    df = filter_data(df, freq=FREQ_REQUIRED)
    df = clean_data(df, keep_flux=keep_flux)
    print(f"[INFO] Dopo pulizia: {df.shape[0]} righe, {df.shape[1]} colonne")

   
    train_df, val_df, test_df = split_data(df, train_ratio, val_ratio, shuffle, random_state)
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "split/train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "split/val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "split/test.csv"), index=False)
    print(f"[OK] File salvati in '{output_dir}'")
    print(f"Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Mars radar dataset")
    parser.add_argument("--input", required=True, help="Percorso al CSV di input")
    parser.add_argument("--output_dir", default="data_cleaned", help="Cartella di output")
    parser.add_argument("--remove_flux", action="store_true", help="Rimuove la colonna FM_data_F10_7_index")
    parser.add_argument("--train_split", type=float, default=0.7, help="Percentuale training (default=0.7)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Percentuale validation (default=0.15)")
    parser.add_argument(
        "--shuffle",
        choices=["none", "all", "train_only"],
        default="none",
        help="Opzione shuffle: none, all o train_only",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Seed per lo shuffle (default=42)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        keep_flux=not args.remove_flux,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        shuffle=args.shuffle,
        random_state=args.random_state,
        
    )




