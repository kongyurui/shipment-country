from typing import Optional

import pandas as pd
import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

from predict import Predictor

HOST_ADDRESS = '0.0.0.0'
HOST_PORT = 8000
TIMEOUT = 60

app = FastAPI(
    title="Shipment Country of Origin Predictor",
    version="1.0.0",
    docs_url="/",
    openapi_url="/api/v1/openapi.json"
)

with open("model_version.txt") as version_fd:
    model_version = version_fd.read()

predictor = Predictor()
predictor.load_model(model_version)

class ShipmentModel(BaseModel):
    arrival_date: str = Body(..., alias="arrivalDate", description="Arrival date for shipment", example="09/20/2012")
    weight_kg: float = Body(None, alias="weightKg", description="Weight of shipment in kilograms", example=7660)
    us_port: Optional[str] = Body(None, alias="usPort", description="Arrival port for shipment", example="Long Beach, California")
    product_details: Optional[str] = Body(None, alias="productDetails", description="Details of shipment product",
                                          example="LA DEE DA ENDCAPLA DEE DA ENDCAPP.O.NO.:0253287160ITEM NO.550808586QTY:6300PCS1CTN63PCSPLACE OF DELIVERY:SOUTHGATE-FLOWGLN: 0078742000008DEPARTMENT NO.: 00007HTS:9503000073,9503000073PO TYPE:0043")


@app.post("/shipments")
def predict_country(shipment: ShipmentModel):

    features = {
        "ARRIVAL.DATE": [shipment.arrival_date],
        "PRODUCT.DETAILS": [shipment.product_details],
        "US.PORT": [shipment.us_port],
        "WEIGHT..KG.": [shipment.weight_kg]
    }
    features_df = pd.DataFrame(features)
    features_df['ARRIVAL.DATE.PROCESSED'] = pd.to_datetime(features_df['ARRIVAL.DATE'])

    country_ids = predictor.predict(features_df)
    country_labels = predictor.label_encoder.inverse_transform(country_ids)

    return {"country": country_labels[0]}


if __name__ == '__main__':
    uvicorn.run(app, host=HOST_ADDRESS, port=HOST_PORT, access_log=False,
                timeout_keep_alive=TIMEOUT)
