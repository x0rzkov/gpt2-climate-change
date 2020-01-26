mkdir -p output
twint -s "#FastFashion OR #ClimateDenier OR #DroughtShaming OR #NoKXL OR #VerticalFarming OR #NotAScientist OR #SaveThePlanetIn4Words OR #BearSelfies OR #BisonSelfies OR #HeatWave" \
      -o "./output/output_en-12.csv" --lang en --csv --location --hide-output
echo 'DONE'
#READ
