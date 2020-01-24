mkdir -p output
twint -s "#peopleofclimate OR #actonclimate OR #climatesmart OR #startseeingco2 OR #climateaction OR #AGW OR #netzero OR #strike4climate OR #climatechange" \
      -o "./output/output_en-4.csv" --lang en --csv --location --hide-output
echo 'DONE'
READ
