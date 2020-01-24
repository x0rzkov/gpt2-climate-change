#!/bin/bash

for i in {0..12}
do
   echo "nohup ./twitter_scrapers/scrub_$i.sh > ./logs/scrub_$i.log &"
   # nohup "./twitter_scrapers/scrub_$i.sh" > "./logs/scrub_$i.log" &
done
