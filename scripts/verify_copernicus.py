"""Explicit live check: query a short reanalysis interval for the example point."""
from datetime import datetime, timezone
from src.copernicus_client import fetch_ssh
from config.settings import REANALYSIS_DATASET

if __name__ == "__main__":
    start=datetime(2026,6,10,tzinfo=timezone.utc)
    end=datetime(2026,6,20,tzinfo=timezone.utc)
    frame,location,metadata=fetch_ssh(12.0,88.0,start,end,REANALYSIS_DATASET)
    print("Dataset:",metadata["dataset_id"])
    print("Actual grid cell:",location)
    print("Variable:",metadata["variable"],"units:",metadata["units"],"representation:",metadata["temporal_representation"])
    print(frame.to_string(index=False))
