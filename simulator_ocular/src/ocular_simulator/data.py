import pandas as pd

def read_csv(filepath):
	df = pd.read_csv(filepath)
	data = df.to_dict(orient='records')
	return data
