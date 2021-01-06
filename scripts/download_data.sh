cd ../data/docvqa/
gsutil -m cp -n gs://neuralresearcher_data/unicamp/ia376j_2020s2/aula10/* .
tar -xf train.tar.gz
tar -xf val.tar.gz
tar -xf test.tar.gz
