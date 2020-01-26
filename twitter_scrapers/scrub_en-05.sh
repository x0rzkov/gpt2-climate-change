mkdir -p output
twint -s "#climateservices OR #greenhypocrisy OR climate change OR #climatehealth OR #climatechanged OR #climateadaptation OR #climatemarxism OR #climateresilience OR #climatehustle" \
      -o "./output/output_en-5.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
