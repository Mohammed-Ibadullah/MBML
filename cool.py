import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Load the dataset
df = pd.read_pickle('merged_df_DK2.pkl')
df['HourDK'] = pd.to_datetime(df['HourDK'])
df = df.drop(['HourUTC', 'PriceArea'], axis=1)
df['hour'] = df['HourDK'].dt.hour

df_use = df[['HourDK', 'Wind', 'Solar', 'Power', 'GrossConsumptionMWh',
             'SpotPriceDKK', 'hourly_temperature_2m', 'cloud_cover', 'hourly_wind_speed_10m', 'hour']]

num_test_days = 4
num_train_days = 1

# Filter data for the first 4 days
start_date = df_use['HourDK'].min()
end_date = start_date + pd.Timedelta(days=num_train_days + num_test_days)
df_filtered = df_use[(df_use['HourDK'] >= start_date) & (df_use['HourDK'] < end_date)]

# Split into training (first 3 days) and testing (4th day)
train_data = df_filtered[df_filtered['HourDK'] < start_date + pd.Timedelta(days=num_test_days)].copy()
test_data = df_filtered[(df_filtered['HourDK'] >= start_date + pd.Timedelta(days=num_test_days)) & (df_filtered['HourDK'] < start_date + pd.Timedelta(days=(num_train_days + num_test_days)))].copy()

# Columns to standardize
columns_to_standardize = ['Wind', 'Solar', 'Power', 'GrossConsumptionMWh',
                          'SpotPriceDKK', 'hourly_temperature_2m', 'cloud_cover', 'hourly_wind_speed_10m']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
train_data[columns_to_standardize] = scaler.fit_transform(train_data[columns_to_standardize])
test_data[columns_to_standardize] = scaler.transform(test_data[columns_to_standardize])

# Define the input features (X) and target variable (y)
X_train = train_data[['hour', 'Wind', 'Solar', 'Power', 'SpotPriceDKK', 'hourly_temperature_2m', 'cloud_cover', 'hourly_wind_speed_10m']].values
y_train = train_data['GrossConsumptionMWh'].values

X_test = test_data[['hour', 'Wind', 'Solar', 'Power', 'SpotPriceDKK', 'hourly_temperature_2m', 'cloud_cover', 'hourly_wind_speed_10m']].values
y_test = test_data['GrossConsumptionMWh'].values

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)

X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Define the neural network model
class FFNN_interpretable(PyroModule):
    def __init__(self, n_in, n_hidden, n_out):
        super(FFNN_interpretable, self).__init__()
        
        # Architecture
        self.in_layer = PyroModule[nn.Linear](n_in, n_hidden)
        self.in_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_in]).to_event(2))

        self.h_layer = PyroModule[nn.Linear](n_hidden, n_hidden)
        self.h_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_hidden]).to_event(2))

        self.out_layer = PyroModule[nn.Linear](n_hidden, n_out)
        self.out_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_out, n_hidden]).to_event(2))

        # Activation functions
        self.tanh = nn.Tanh()
        
    def forward(self, X, y=None):
        X_nn = X[:, 1:]  # Skip the first column for the neural network part
        X_nn = self.tanh(self.in_layer(X_nn))
        X_nn = self.tanh(self.h_layer(X_nn))
        X_nn = self.out_layer(X_nn)
        nn_out = X_nn.squeeze(-1)

        beta_lin = pyro.sample("beta", dist.Normal(0, 1))
        X_linear = X[:, 0]  # Use the first column for the linear part
        with pyro.plate("observations", X.shape[0]):
            linear_out = X_linear * beta_lin
            y = pyro.sample("obs", dist.Normal(nn_out + linear_out, 0.1), obs=y)
            
        return y

# Initialize the model, guide, and optimizer
model = FFNN_interpretable(n_in=X_train.shape[1], n_hidden=4, n_out=1)
guide = AutoDiagonalNormal(model)

# Reset parameter values
pyro.clear_param_store()

# Define the number of optimization steps
n_steps = 10000

# Setup the optimizer
adam_params = {"lr": 0.01}
optimizer = ClippedAdam(adam_params)

# Setup the inference algorithm
elbo = Trace_ELBO(num_particles=1)
svi = SVI(model, guide, optimizer, loss=elbo)

# Training loop
for step in range(n_steps):
    elbo = svi.step(X_train_tensor, y_train_tensor)
    if step % 500 == 0:
        print(f"[{step}] ELBO: {elbo:.1f}")

# After training, you can use the model and guide to make predictions on the test data
