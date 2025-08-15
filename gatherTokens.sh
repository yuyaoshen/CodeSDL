#!/bin/bash

path=./data/BCB
folder=basic
rm $path/doc_"$folder".txt

ls $path/dataset_$folder | while read filename
do
	cat $path/dataset_$folder/"$filename" >> $path/doc_"$folder".txt
done
