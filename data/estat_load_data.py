import requests
import pandas as pd
import sys
import json
from IPython.display import display


# ----------------------------------------------------------- #
# ---  Main project file for loading and preaparing data  --- #
# ---  (set up after testing the functionalities in Notebook) #
# ----------------------------------------------------------- #

# URL of the JSON data
url_old = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/tour_occ_nim?format=JSON&sinceTimePeriod=2012-01&geo=DK&geo=DE&geo=EL&geo=ES&geo=HR&geo=IT&geo=PT&geo=FI&geo=SE&geo=NO&unit=NR&unit=PCH_SM&unit=PCH_SM_19&c_resid=DOM&c_resid=FOR&nace_r2=I551&nace_r2=I552&nace_r2=I553&lang=de"

url_domain = "https://ec.europa.eu/eurostat/"
url_site = "api/dissemination/statistics/1.0/data/tour_occ_nim"
url_qry_base = "?format=JSON"
url_qry_period_from = "&sinceTimePeriod=2012-01"
url_qry_period_to = ""
url_qry_geo = "&geo=DK&geo=DE&geo=EL&geo=ES&geo=HR&geo=IT&geo=PT&geo=FI&geo=SE&geo=NO"
url_qry_unit = "&unit=NR&unit=PCH_SM&unit=PCH_SM_19"
url_qry_resid = "&c_resid=DOM&c_resid=FOR"
url_qry_nace = "&nace_r2=I551&nace_r2=I552&nace_r2=I553"
url_qry_lang = "&lang=de"

url = url_domain + url_site + url_qry_base + url_qry_period_from + url_qry_period_to + url_qry_geo + url_qry_unit + url_qry_resid + url_qry_nace + url_qry_lang
if url == url_old:
    print("URL PATH match.")
else:
    print ("ERROR: URL mismatch!")
# sys.exit()

# Download and parse the JSON
response = requests.get(url)
data = response.json()
dims = data['dimension']
values = data['value']

# Extract dimension metadata
dim_order = data['id']  # dimension order
dim_sizes = data['size']  # sizes of dimensions
dim_labels = {dim: dims[dim]['label'] for dim in dim_order}
dim_categories = {dim: dims[dim]['category']['label'] for dim in dim_order}
dim_category_keys = {dim: list(dims[dim]['category']['label'].keys()) for dim in dim_order}

# Print dimension overviews in table format
print("DIMENSION OVERVIEW:\n")
for dim in dim_order:
    label = dim_labels[dim]
    categories = dim_categories[dim]
    
    df = pd.DataFrame(list(categories.items()), columns=[f"{label} ID", f"{label} Label"])
    print(f"Dimension: {label}")
    # print(df.to_string(index=False))
    # display(df) # Use with IPython imported module only
    # print("\n" + "-" * 40 + "\n")


# sys.exit()

# Flatten values
records = []
for idx, val in values.items():
    idx = int(idx)
    indexes = []
    remainder = idx
    for size in reversed(dim_sizes):
        indexes.append(remainder % size)
        remainder //= size
    indexes.reverse()

    # Map to dimension keys and values
    row = {}
    for i, dim in enumerate(dim_order):
        keys = dim_category_keys[dim]
        key = keys[indexes[i]]
        label = dim_categories[dim][key]
        dim_name = dim_labels[dim]

        # Column names
        if dim == 'freq':
            row['Zeitliche_Frequenz_Idx'] = key
            row['Zeitliche_Frequenz'] = label
        elif dim == 'c_resid':
            row['Aufenthaltsland_Idx'] = key
            row['Aufenthaltsland'] = label
        elif dim == 'unit':
            row['Maßeinheit_Idx'] = key
            row['Maßeinheit'] = label
        elif dim == 'nace_r2':
            row['NACEr2_Idx'] = key
            row['NACEr2'] = label
        elif dim == 'geo':
            row['Geopolitische_Meldeeinheit_Idx'] = key
            row['Geopolitische_Meldeeinheit'] = label
        elif dim == 'time':
            # row['Zeit_Idx'] = key
            row['JahrMonat'] = label
    row['value'] = val
    records.append(row)

# Create DataFrame
df = pd.DataFrame(records)

# Export to CSV
# csv_filename = "estat_tour_overnight_stays_2012-2025_eu10_de.csv"
# df.to_csv(csv_filename, index=False)
# print(f"Exported to {csv_filename}")
print(df.tail(30))
