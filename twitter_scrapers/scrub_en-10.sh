mkdir -p output
twint -s "#ActOnClimate OR #BlackFridayin3Words OR #OptOutside OR #Go100Percent OR #Renewables OR #ElNino OR #ShellNo OR #DumpTrump OR #FeelTheBern OR #TPP OR #BanTheBead" \
      -o "./output/output_en-10.csv" --lang en --csv --location --hide-output
echo 'DONE'
READ
