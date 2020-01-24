mkdir -p output
twint -s "#العمل_المناخي OR #تغيرـالمناخ" \
      -o "./output/output_ar-00.csv" --lang ar --csv --location --hide-output
echo 'DONE'
READ
