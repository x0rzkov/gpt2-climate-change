mkdir -p output
twint -s "#climate OR #climateemergency OR #climatechangeisreal OR #oceanforclimate OR #gretathunberg OR #climatehope OR #climatejustice OR #youthstrike4climate OR #climatebrawl" \
      -o "./output/output_en-3.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
