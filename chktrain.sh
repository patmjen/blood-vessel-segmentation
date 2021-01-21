#!/bin/bash

for id in $(bjobs -noheader -ojobid); do 
	echo -n "Job $id: "
	bpeek $id | grep "val_loss reached" | tail -n1
done
