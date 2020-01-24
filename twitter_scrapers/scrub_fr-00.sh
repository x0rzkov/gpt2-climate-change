mkdir -p output
twint -s "#réchauffementclimatique OR #environnement OR #transitionzerocarbone #transitionécologique OR #ecologie #climat OR #ecologie OR #carbone" \
      -o "./output/output_fr-00.csv" --lang fr --csv --location --hide-output
echo 'DONE'
READ

# nohup ./twitter_scrapers/scrub_fr-00.sh > ./logs/scrub_fr-00.log &