mkdir -p output
twint -s "" \
      -o "./output/output_0-fr.csv" --lang fr --csv --location --hide-output
echo 'DONE'
READ
