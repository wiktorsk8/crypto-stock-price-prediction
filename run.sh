#!/bin/bash

docker build -t stock-price-prediction .
docker run --name stock-prediction-container --network host stock-price-prediction
