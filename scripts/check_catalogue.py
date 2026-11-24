"""Verify product dataset and zos metadata from the live Copernicus catalogue."""
import copernicusmarine
from config.settings import (REANALYSIS_PRODUCT,REANALYSIS_DATASET,
                             OPERATIONAL_PRODUCT,OPERATIONAL_DATASET,SSH_VARIABLE)

for product_id,dataset_id in ((REANALYSIS_PRODUCT,REANALYSIS_DATASET),(OPERATIONAL_PRODUCT,OPERATIONAL_DATASET)):
    catalogue=copernicusmarine.describe(product_id=product_id,disable_progress_bar=True)
    dataset=next((item for item in catalogue.products[0].datasets if item.dataset_id==dataset_id),None)
    if dataset is None: raise SystemExit(f"Dataset {dataset_id} not listed under {product_id}")
    service=next((s for v in dataset.versions for p in v.parts for s in p.services if s.service_short_name in ("arco-time-series","arco-geo-series","geoseries")),None)
    services=[s for v in dataset.versions for p in v.parts for s in p.services]
    variable=next((var for s in services for var in s.variables if var.short_name==SSH_VARIABLE),None)
    if variable is None: raise SystemExit(f"{dataset_id} does not advertise {SSH_VARIABLE}")
    times=[c for c in variable.coordinates if c.coordinate_id in ("time","time_counter")]
    print(product_id)
    print(" dataset_id:",dataset.dataset_id)
    print(" variable:",variable.short_name,"standard_name:",variable.standard_name,"units:",variable.units)
    if times: print(" time coordinate coverage (UTC epoch ms):",times[0].minimum_value,"to",times[0].maximum_value,"step:",times[0].step)
