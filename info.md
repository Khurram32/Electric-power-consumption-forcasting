# Electric-power-consumption-forcasting
This contains a code written in python for electric power consumption forcasting which uses 5 algorithms which are SVM, Linear Regression, decision Tree, Random Forest, k-NN. 
dataset for this project is taken form kaggle and contain 52,000+ records of electric onsumption data.
The main takeaway from this session and the examination of the given graph highlights the
efficiency and relative effectiveness of different machine learning algorithms in predicting
electric power usage. Among the models, Random Forest and Decision Tree closely mirror the
real consumption trends, effectively capturing the dynamic highs and lows, showcasing their
capability to model nonlinear relationships and significant fluctuations in time-series data. The
k-Nearest Neighbors (k-NN) model shows decent performance, though it has a more gradual
output and demonstrates some delay when reacting to rapid changes. Conversely, Linear
Regression and Support Vector Machine (SVM) generate fairly flat or less sensitive forecasts,
highlighting their shortcomings in managing intricate, nonlinear, and highly variable
consumption patterns. Significantly, post mid-March, all models appear to generate static or
extrapolated results, indicating that their training datasets may have been restricted to prior
dates and lack the ability to generalize beyond that timeframe. This conduct underscores the
necessity for regular retraining or continuous updates to uphold prediction precision over time.
The outcome derived from this analysis indicates that employing a mix of algorithms offers a
comprehensive perspective on model performance, helps in choosing the most precise and
reliable method for implementation, and guarantees enhanced resilience in predictions. In
particular, ensemble techniques such as Random Forest show enhanced effectiveness in
managing unusual consumption trends, rendering them ideal for real-world electric load
forecasting tasks. This multi-model strategy enhances reliability and reduces the risks linked
to depending on a single predictive model in changing energy settings.
