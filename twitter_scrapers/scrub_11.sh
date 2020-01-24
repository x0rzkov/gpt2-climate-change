mkdir -p output
twint -s "#renewableenergy OR #peace OR #sustainablefashion OR #vegan OR #zerowaste OR #plasticfree OR #sustainableliving OR #globalwarmingisreal" \
      -o "./output/output_11.csv" --lang en --csv --location --hide-output
echo 'DONE'
READ
