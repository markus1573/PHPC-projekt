import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('full_results.csv')

plt.figure(figsize=(10, 6))
plt.hist(df[' mean_temp'], bins=50, edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('mean_temp_histogram.png')
print("a) Histogram saved as 'mean_temp_histogram.png'")

avg_mean_temp = df[' mean_temp'].mean()
print(f"b) Average mean temperature: {avg_mean_temp:.2f} °C")

avg_std_temp = df[' std_temp'].mean()
print(f"c) Average temperature standard deviation: {avg_std_temp:.2f} °C")

count_above_18 = (df[' pct_above_18'] >= 50).sum()
print(f"d) Buildings with at least 50% area above 18ºC: {count_above_18}")

count_below_15 = (df[' pct_below_15'] >= 50).sum()
print(f"e) Buildings with at least 50% area below 15ºC: {count_below_15}")
