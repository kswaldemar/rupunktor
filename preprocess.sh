#!/bin/bash
FNAME=$1

grep -v '\"' ${FNAME} | egrep -v '[-:;!?()]' | grep '\.$' > raw_text.txt

