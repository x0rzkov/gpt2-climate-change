mkdir -p output
twint -s "#climateurgency OR #climatesilence OR #cleanenergyfuture OR #peoplesclimate OR #climatedisruption OR #naturalclimatesolutions OR #climatebudget OR #riseforclimate OR #endclimatesilence" \
      -o "./output/output_en-1.csv" --lang en --csv --location --hide-output
echo 'DONE'
# READ
