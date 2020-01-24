mkdir -p output
twint -s "#scioclimate OR #climatescam OR #stateofclimate OR #climateactionnow OR #climatecrisis OR #climatetownhall OR #climatefact OR #climaterisk OR #climatechangeshealth" \
      -o "./output/output_6.csv" --lang en --csv --location --hide-output
echo 'DONE'
READ
