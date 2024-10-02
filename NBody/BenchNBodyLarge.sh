#!/bin/bash
DEVICE=0
ITERATIONS=100
END=20
[ ! -z "$1" ] && DEVICE="$1"
# Test 32 bits
seq 5 1 $END | while read I; do
    echo -ne "$((2**$I)) "
    python3 NBody.py -d $DEVICE -n $((2**$I)) -i $ITERATIONS -s 0.1 -m ImplicitEuler -t FP32 2>/dev/null | egrep 'Median' | awk '{ print $NF }' | head -1
done
echo
# Test 64 bits
ITERATIONS=10
seq 5 1 $END | while read I; do
    echo -ne "$((2**$I)) " ;
    python3 NBody.py -d $DEVICE -n $((2**$I)) -i $ITERATIONS -s 0.1 -m ImplicitEuler -t FP64 2>/dev/null | egrep 'Median' | awk '{ print $NF }' | head -1
done


