mkdir -p output
twint -s "#climatemarch OR #globalclimatestrike OR #climat OR #environmentaljustice OR #climatesolutions OR #nofossilfuelmoney OR #climatehkh OR #parisagreement OR global warming" \
      -o "./output/output_en-2.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
