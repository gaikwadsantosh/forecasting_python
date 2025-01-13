

#To run streamlit code directly from prompt, you get benefit of hot reload on code changes.
streamlit run main.py

NOTE: To build and run docker image, please logout from VPN, as it would restrict connection to public docker images

#Build docker image
docker build -t streamlitucmforecast:v1 .

#Run Docker image
docker run -p 8501:8501 streamlitucmforecast:v1
NOTE: here docker port 8501 is mapped to host machine 8501, hence on host machine use http://localhost:8501

#Run with Docker Compose
docker-compose up
NOTE: here docker port 8501 is mapped to host machine 8081, hence on host machine use http://localhost:8081


#local Postgre image
docker run --name my_postgres -e POSTGRES_PASSWORD=MyPass -d -p 5432:5432 postgres
#connect to local postgre
psql -h localhost -p 5432 -U postgres (postgres is username here and 5432 is port)
SELECT datname FROM pg_database;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
##sample docker commands
docker ps -a
docker start my_postgres

CREATE TABLE IF NOT EXISTS dbo.medicalcount (
    monthdate DATE NOT NULL,
    valuecount INTEGER NOT NULL
);
select * from dbo.medicalcount limit 10;


-----------------
Deployment to ACR
-----------------
Run following commands from command prompt
1. docker build -t medicalforecast-streamlit:v2 .
2. docker tag medicalforecast-streamlit:v2 testacr.azurecr.io/medicalforecast-streamlit:v2
3. either use docker login to ACR and use Docker Push command to push image to ACR or use VSCode Extensions to push image to ACR
4. go to deployment center of Azure web app and change the image details (e.g. version), then wait for few mins for image deployment.
---------------
Azure Resource
---------------- 
Azure Web App: forecastappai
Azure Resource Group: txcCosmosEvaluation

---------------------------------
---------Model Selection---------
---------------------------------
#Use Facebook Prophet when:
Your time-series data has multiple seasonality patterns (like daily, weekly, yearly).
You have known events or holidays which influence the time series.
You want to handle outliers, missing values, and shifts in trend within the model.
You need a model that is easy to implement, interpret, and tweak.
Your data is non-stationary and exhibits complex patterns.

#Use ARIMA when:
Your data shows a clear trend and/or seasonality.
You have a linear relationship in your time series data.
Your data is stationary or can be easily transformed into a stationary form.

#Use UCM when:
You have multiple component trends (e.g., seasonal, cyclical, irregular) that need to be isolated and analyzed separately.
You need to model time series data that may have non-linear patterns.
Your data is non-stationary and it's hard to transform it into a stationary form.

#Use Holt-Winters when:
Our Time series data has clear trend and seasonality.
The trend and seasonal components are relatively stable and do not vary with the level of the time series.
We require short-term forecasts for near-future predictions.
---------------------------------
