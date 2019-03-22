#!/bin/bash
CLASS_FILE=classes.txt
SRC=/path/to/modelnet40v1png
TARGET=/path/to/modelnet40v1

id=0
while read line
do
	echo $id, $line
	for stage in "train" "test"
	do
		baseDir=$stage/$line
		mkdir -p $baseDir
		list=$(find $SRC/$line/$stage/ -name *.off -exec basename {} \;)
		for img in $list
		do
			tag=${img%.off}
			view=($(ls $TARGET/$line/$stage/${tag}*.png))
			fname=$baseDir/$(basename $img).txt

			touch $fname
			echo $id >> $fname
			echo "${#view[@]}" >> $fname
			for elem in ${view[@]}
			do
				echo $elem >> $fname
			done
		done
	done

	let "id = id+1"
done < $CLASS_FILE
