mkdir -p output
twint -s "#climatetutorial OR #climatemigration OR #up4climate OR #youthforclimate OR #climateinsurance OR #coveringclimatenow OR #climatefriday OR #climatefinance OR #carbonbudget" \
      -o "./output/output_en-7.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
