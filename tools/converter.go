package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/k0kubun/pp"
	"gopkg.in/yaml.v3"
)

type Occurence struct {
	Data []Datum `json:"data,omitempty"`
}

type Datum struct {
	Source  string `json:"source,omitempty"`
	Target  string `json:"target,omitempty"`
	TweetID string `json:"tweet_id,omitempty"`
	Type    string `json:"type,omitempty"`
	Weight  string `json:"weight,omitempty"`
}

func main() {

	filePath := "../misc/co-occur_fr.yml"
	yamlFile, err := os.Open(filePath)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened file: ", filePath)
	// defer the closing of our jsonFile so that we can parse it later on
	defer yamlFile.Close()

	byteValue, _ := ioutil.ReadAll(yamlFile)

	var occurence Occurence
	err = yaml.Unmarshal(byteValue, &occurence)
	if err != nil {
		log.Fatalf("cannot unmarshal data: %v\n", err)
	}

	for _, occur := range occurence.Data {
		pp.Println(occur.Target, occur.Source)
	}
}
