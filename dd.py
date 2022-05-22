arima_model.plot_diagnostics(figsize=(7,5))
plt.show()

# Forecast
n_periods = len(test)
fc, confint = arima_model.predict(n_periods=n_periods, return_conf_int=True)


index_of_fc = test.index

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
# plt.plot(train)
plt.plot(test, color="blue")
plt.plot(fc_series, color='orange')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast")
plt.show()