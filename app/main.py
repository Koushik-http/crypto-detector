import os
import yaml
import logging.config
from fastapi import FastAPI, HTTPException, Depends
from app.api import endpoints
from app.db.database import engine, Base
from app.services.fraud_detection_service import FraudDetectionService
from config.settings import settings
from app.db.database import Base, engine

def setup_logging():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'logging_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

def create_app() -> FastAPI:
    setup_logging()
    
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG_MODE,
    )

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Include API routes
    app.include_router(endpoints.router, prefix=settings.API_V1_STR)

    return app

app = create_app()

# Dependency to get the fraud detection service
def get_fraud_detection_service():
    return FraudDetectionService()

@app.get("/")
async def root():
    return {"message": "Welcome to the Cryptocurrency Fraud Detection API"}

@app.post("/predict")
async def predict_fraud(transaction: dict, service: FraudDetectionService = Depends(get_fraud_detection_service)):
    try:
        prediction = service.predict(transaction)
        return {"prediction": prediction}
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)