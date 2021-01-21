#!/bin/bash

for f in $(ls -1 batch_output | grep train_.*err); do
	echo -n "$f: "
	grep "val_loss reached" batch_output/$f | tail -n1
done
