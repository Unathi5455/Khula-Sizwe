Verification Code: WTC-R6BV6YJP

Drought Early-Warning System for Food Security in Africa

An automated data engineering pipeline that ingests climate and soil data to forecast drought conditions and support early intervention against food insecurity in vulnerable African regions.

Overview

Sub-Saharan Africa accounts for over 60% of global drought-related food insecurity events. Traditional monitoring is reactive and fragmented.

Khula-Sizwe is an automated, scalable data engineering pipeline and early-warning system that ingests multi-source climate and soil data, processes it in near-real-time, forecasts drought probability using machine learning, and triggers actionable alerts for vulnerable communities, NGOs, and policymakers

Problem Statement

Droughts in Sub-Saharan Africa are a leading driver of crop failure and food insecurity, but early-warning information is often fragmented, delayed, or inaccessible to the communities and decision-makers who need it most. This pipeline addresses that gap by automating the collection and analysis of publicly available climate and soil datasets to produce timely, region-level drought risk forecasts.


Objectives
Automate ingestion of climate and soil data for target African regions
Engineer drought-relevant features (e.g. rainfall anomalies, soil moisture deficits, vegetation indices)
Train and evaluate forecasting models to predict drought onset/severity
Surface predictions through an accessible early-warning dashboard
Support downstream decision-making for food security mitigation

Pipeline Architecture

Raw Data (APIs / satellite feeds)
        │
        ▼
  Ingestion Layer  ──► Data Lake (raw)
        │
        ▼
 Cleaning & Validation
        │
        ▼
Feature Engineering  ──► Data Warehouse (processed)
        │
        ▼
  Forecasting Model
        │
        ▼
Early-Warning Dashboard / Alerts

Each stage is designed to run independently and be orchestrated on a schedule, allowing the system to refresh forecasts as new climate data becomes available.

Tech Stack

Layer	Tools
Orchestration	Apache Airflow / Prefect
Ingestion	Python, REST APIs, requests, xarray
Storage	PostgreSQL / Parquet on cloud storage
Processing	pandas, NumPy, xarray, GeoPandas
Modeling	scikit-learn, LightGBM
Visualization	Plotly Dash
Deployment	Docker

Project Structure

├── data/
│   ├── raw/                # Unprocessed source data
│   └── processed/          # Cleaned, feature-engineered data
├── src/
│   ├── ingestion/          # Data collection scripts
│   ├── processing/         # Cleaning & feature engineering
│   ├── modeling/           # Model training & evaluation
│   └── dashboard/          # Early-warning visualization app
├── notebooks/               # Exploratory analysis
├── config/                 # Region and data source configuration
├── tests/                  # Unit and pipeline tests
├── requirements.txt
└── README.md
