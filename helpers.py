import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot1(merged_df_DK2):
    plt.figure(figsize=(12, 8))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Set colors for plots
    colors = sns.color_palette("husl", 5)

    # Define hours of the day
    hours = np.arange(1, 25)

    # Calculate average energy and standard deviation for each hour
    wind_avg = []
    wind_std = []
    solar_avg = []
    solar_std = []
    power_avg = []
    power_std = []

    for hour in hours:
        wind_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Wind']
        solar_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Solar']
        power_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Power']

        wind_avg.append(wind_hourly.mean())
        wind_std.append(wind_hourly.std())

        solar_avg.append(solar_hourly.mean())
        solar_std.append(solar_hourly.std())

        power_avg.append(power_hourly.mean())
        power_std.append(power_hourly.std())

    # Plot Wind, Solar, and Power
    plt.subplot(3, 2, 1)
    plt.plot(merged_df_DK2['HourDK'], merged_df_DK2['Wind'], color=colors[0], label='Line Plot')
    plt.title('Wind')
    plt.xlabel('Time')
    plt.ylabel('Wind Energy (MWh)')

    plt.subplot(3, 2, 2)
    sns.violinplot(x=merged_df_DK2['Wind'], color=colors[0])
    plt.title('Wind')
    plt.xlabel('Wind Energy (MWh)')
    plt.ylabel('Density')

    plt.subplot(3, 2, 3)
    plt.plot(merged_df_DK2['HourDK'], merged_df_DK2['Solar'], color=colors[1], label='Line Plot')
    plt.title('Solar')
    plt.xlabel('Time')
    plt.ylabel('Solar Energy (MWh)')

    plt.subplot(3, 2, 4)
    sns.violinplot(x=merged_df_DK2['Solar'], color=colors[1])
    plt.title('Solar')
    plt.xlabel('Solar Energy (MWh)')
    plt.ylabel('Density')

    plt.subplot(3, 2, 5)
    plt.plot(merged_df_DK2['HourDK'], merged_df_DK2['Power'], color=colors[2], label='Line Plot')
    plt.title('Power')
    plt.xlabel('Time')
    plt.ylabel('Power (MWh)')

    plt.subplot(3, 2, 6)
    sns.violinplot(x=merged_df_DK2['Power'], color=colors[2])
    plt.title('Power')
    plt.xlabel('Power (MWh)')
    plt.ylabel('Density')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

    # Subplot for Hydro and GridLoss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(merged_df_DK2['HourDK'], merged_df_DK2['Hydro'], color=colors[3], label='Hydro')
    plt.title('Hydro')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Hydro Energy (MWh)')

    plt.subplot(1, 2, 2)
    plt.plot(merged_df_DK2['HourDK'], merged_df_DK2['GridLoss'], color=colors[4], label='GridLoss')
    plt.title('GridLoss')
    plt.xlabel('Hour of the Day')
    plt.ylabel('GridLoss (MWh)')

    plt.tight_layout()
    plt.show()



def plot2(merged_df_DK2):
    import numpy as np

    plt.figure(figsize=(12, 10))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define hours of the day
    hours = np.arange(1, 25)

    # Calculate average energy and standard deviation for each hour
    wind_avg = []
    wind_std = []
    solar_avg = []
    solar_std = []
    power_avg = []
    power_std = []
    consumption_avg = []
    consumption_std = []

    for hour in hours:
        wind_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Wind']
        solar_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Solar']
        power_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['Power']
        consumption_hourly = merged_df_DK2[merged_df_DK2['HourDK'].dt.hour == hour]['GrossConsumptionMWh']
        
        wind_avg.append(wind_hourly.mean())
        wind_std.append(wind_hourly.std())
        
        solar_avg.append(solar_hourly.mean())
        solar_std.append(solar_hourly.std())
        
        power_avg.append(power_hourly.mean())
        power_std.append(power_hourly.std())
        
        consumption_avg.append(consumption_hourly.mean())
        consumption_std.append(consumption_hourly.std())

    # Plot Wind
    plt.subplot(4, 1, 1)
    plt.plot(hours, wind_avg, color='blue', label='Average Wind Energy')
    plt.fill_between(hours, np.array(wind_avg) - np.array(wind_std), np.array(wind_avg) + np.array(wind_std), color='lightblue', alpha=0.5, label='Deviation')
    plt.title('Hourly Development of Wind Energy')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Wind Energy (MWh)')
    plt.legend()

    # Plot Solar
    plt.subplot(4, 1, 2)
    plt.plot(hours, solar_avg, color='orange', label='Average Solar Energy')
    plt.fill_between(hours, np.array(solar_avg) - np.array(solar_std), np.array(solar_avg) + np.array(solar_std), color='lightsalmon', alpha=0.5, label='Deviation')
    plt.title('Hourly Development of Solar Energy')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Solar Energy (MWh)')
    plt.legend()

    # Plot Power
    plt.subplot(4, 1, 3)
    plt.plot(hours, power_avg, color='green', label='Average Power')
    plt.fill_between(hours, np.array(power_avg) - np.array(power_std), np.array(power_avg) + np.array(power_std), color='lightgreen', alpha=0.5, label='Deviation')
    plt.title('Hourly Development of Power')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Power (MWh)')
    plt.legend()

    # Plot Gross Consumption
    plt.subplot(4, 1, 4)
    plt.plot(hours, consumption_avg, color='red', label='Average Gross Consumption')
    plt.fill_between(hours, np.array(consumption_avg) - np.array(consumption_std), np.array(consumption_avg) + np.array(consumption_std), color='lightcoral', alpha=0.5, label='Deviation')
    plt.title('Hourly Development of Gross Consumption')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Gross Consumption (MWh)')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def plot3(merged_df_DK1):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a figure and axes with subplots
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define variables
    variables = ['hourly_temperature_2m', 'precipitation', 'cloud_cover', 'hourly_wind_speed_10m']

    # Plot line plot on the left and box plot on the right for each variable
    for i, variable in enumerate(variables):
        if variable == 'cloud_cover':
            hourly_bins = merged_df_DK1.groupby(merged_df_DK1['HourDK'].dt.hour)[variable].mean()
            axs[i, 0].plot(hourly_bins.index, hourly_bins, color='blue', marker='o')
            axs[i, 0].set_title(f'{variable} (Line Plot)')
        else:
            axs[i, 0].plot(merged_df_DK1['HourDK'], merged_df_DK1[variable], color='blue')
            axs[i, 0].set_title(f'{variable} (Line Plot)')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel(variable)
        
        if variable == 'precipitation':
            sns.boxplot(x=merged_df_DK1[variable], ax=axs[i, 1], color='orange')
            axs[i, 1].set_title(f'{variable} (Box Plot)')
            #this took too long ;(
        else:
            sns.violinplot(x=merged_df_DK1[variable], ax=axs[i, 1], color='orange')
            axs[i, 1].set_title(f'{variable} (Violin Plot)')
        axs[i, 1].set_xlabel(variable)
        axs[i, 1].set_ylabel('Density' if variable != 'precipitation' else 'Value')

    plt.tight_layout()
    plt.show()


def cor_plot(merged_df_DK2):
    correlation_matrix = merged_df_DK2[['Wind', 'Solar', 'Power', 'hourly_temperature_2m', 'precipitation', 'cloud_cover', 'hourly_wind_speed_10m','GrossConsumptionMWh']].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    weather_features = ['hourly_temperature_2m', 'precipitation', 'cloud_cover', 'hourly_wind_speed_10m']
    ax = plt.gca()
    ax.set_xticklabels([f'$\mathbf{{{label}}}$' if label in weather_features else f'${label}$' for label in correlation_matrix.columns])
    ax.set_yticklabels([f'$\mathbf{{{label}}}$' if label in weather_features else f'${label}$' for label in correlation_matrix.index])

    plt.title('Correlation Heatmap')
    plt.show()
