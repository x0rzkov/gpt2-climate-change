#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np
import re

data = pd.read_csv('path to file created with consolidate_clean_data.py')
print(len(data))

# Filter by location
US_data = data[data['location'].str.contains('Alabama|AL|Alaska|AK|Arizona|AZ|Arkansas|AR|California|CA|Colorado|CO|Connecticut|CT|Delaware|DE|Florida|FL|Georgia|GA|Hawaii|HI|Idaho|ID|Illinois|IL|Indiana|IN|Iowa|IA|Kansas|KS|Kentucky|KY|Louisiana|LA|Maine|ME|Maryland|MD|Massachusetts|MA|Michigan|MI|Minnesota|MN|Mississippi|MS|Missouri|MO|Montana|MT|Nebraska|NE|Nevada|NV|New Hampshire|NH|New Jersey|NJ|New Mexico|NM|New York|NY|North Carolina|NC|North Dakota|ND|Ohio|OH|Oklahoma|OK|Oregon|OR|Pennsylvania|PA|Rhode Island|RI|South Carolina|SC|South Dakota|SD|Tennessee|TN|Texas|TX|Utah|UT|Vermont|VT|Virginia|VA|Washington|WA|West Virginia|WV|Wisconsin|WI|Wyoming|WY')==True]
print(len(US_data))

# In[ ]:
# Process truncated tweets
truncated = US_data.copy()
truncated = truncated[truncated['truncated'] == True]
truncated = truncated[['created_at','timezone','user_id','extended_text', 'location']]
truncated = truncated.rename(columns={'extended_text':'text'})
print(len(truncated))
truncated.head()

# In[ ]:
# Process not truncated tweets
not_trunc = US_data.copy()
not_trunc = not_trunc[not_trunc['truncated'] == False]
not_trunc = not_trunc[not_trunc['RT_truncated'] != False]
not_trunc = not_trunc[not_trunc['RT_truncated'] != True]
not_trunc = not_trunc[['created_at','timezone','user_id','text', 'location']]
print(len(not_trunc))
not_trunc.head()

# In[ ]:
# Process truncated REtweets
RT_truncated = US_data.copy()
RT_truncated = RT_truncated[RT_truncated['RT_truncated'] == True]
RT_truncated = RT_truncated[['created_at','timezone','user_id','extended_RT_text', 'location']]
RT_truncated = RT_truncated.rename(columns={'extended_RT_text':'text'})
print(len(RT_truncated))
RT_truncated.head()

# In[ ]:
# Process not truncated REtweets
RT = US_data.copy()
RT = RT[RT['RT_truncated'] == False]
RT = RT[['created_at','timezone','user_id','RT_text', 'location']]
RT = RT.rename(columns={'RT_text':'text'})
print(len(RT))
RT.head()

# In[ ]:
# Concatenate dataframes to get full tweets data
full = pd.concat([truncated,not_trunc,RT_truncated,RT])
print(len(full))

# In[ ]:
# Clean up to extract tweet from text
full['tweet'] = full['text']
full['tweet'] = full['tweet'].str.replace(r'RT ', '')
full['tweet'] = full['tweet'].str.replace(r'@([^ ]*)', '')
full['tweet'] = full['tweet'].str.replace(r'(?= http).*$', '')
full['tweet'] = full['tweet'].str.strip()

# Get hashtags
full['hashtags'] = full['tweet']
full['hashtags'] = full['hashtags'].apply(lambda x: re.findall(r"#(\w+)", x))

# Get date and time
full['date_time'] = pd.to_datetime(full['created_at'], errors='coerce')
full['date'] = full['date_time'].dt.date
full['time'] = full['date_time'].dt.time

# Sort chronologically
full.sort_values('date_time', inplace = True)

# Keep and order desired columns
full = full[['date', 'time', 'timezone', 'user_id', 'tweet', 'hashtags', 'location']]
full.head()

# In[ ]:
# Export to csv
full.to_csv(r'desired path to final data', encoding='utf-8-sig')

