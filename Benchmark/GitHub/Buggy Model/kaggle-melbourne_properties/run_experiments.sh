#!/bin/bash


# Two FFN layers
for val in {1..50}
do
	python price-keras.py --train --num_epochs 200 --type FFN --model $val,1
done

#Three layers
for fval in {1..50}
do
	for sval in {1..50}
	do
		python price-keras.py --train --num_epochs 200 --type FFN --model $fval,$sval,1
	done
done
# Four layers
