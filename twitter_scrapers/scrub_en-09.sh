mkdir -p output
twint -s "#cleanpowerplan OR #climateliability OR #unclimatesummit OR #climatebreakdown" \
      -o "./output/output_en-9.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
