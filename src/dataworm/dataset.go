package dataworm

import (
	"strconv"
	"bufio"
	"strings"
	"os"
)

type DataSet struct {
	Samples []Sample
}

func (d *DataSet) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
  	for scanner.Scan() {
		line := strings.Replace(scanner.Text(), " ", "\t", -1)
		tks := strings.Split(line, "\t")
		sample := Sample{Features : []Feature{}, Label : 0}
		for i, tk := range tks {
			if i == 0 {
				label, err := strconv.ParseInt(tk, 10, 16)
				if err != nil {
					break
				}
				if label > 0 {
					sample.Label = 1
				} else{
					sample.Label = 0
				}	
			} else{
				kv := strings.Split(tk, ":")
				feature_id, err := strconv.ParseInt(kv[0], 10, 64)
				if err != nil {
					break
				}
				feature_value, err := strconv.ParseFloat(kv[1], 64)
				if err != nil {
					break
				}
				feature := Feature{feature_id, feature_value}
				sample.Features = append(sample.Features, feature)
			}
		}
		d.Samples = append(d.Samples, sample)	
	}
	return scanner.Err()	
}